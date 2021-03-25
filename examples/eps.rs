use rasterize::*;
use std::{
    cell::{self, RefCell},
    cmp::{Ordering, PartialEq},
    collections::HashMap,
    fmt,
    io::Read,
    num::ParseFloatError,
    rc::Rc,
    str::FromStr,
    string::FromUtf8Error,
};

#[derive(Clone)]
struct PSGraphicState {
    color: LinColor,
    miter_limit: Scalar,
    stroke_style: StrokeStyle,
    transform: Transform,
    transform_inverse: Option<Transform>,
    path: PathBuilder,
    clip_path: Path,
}

impl PSGraphicState {
    fn new(transform: Transform) -> Self {
        Self {
            color: LinColor::new(0.0, 0.0, 0.0, 1.0),
            miter_limit: 10.0,
            stroke_style: StrokeStyle {
                width: 1.0,
                line_join: LineJoin::Miter(10.0),
                line_cap: LineCap::Butt,
            },
            transform,
            transform_inverse: None,
            path: PathBuilder::new(),
            clip_path: Path::empty(),
        }
    }

    fn transform_set(&mut self, tr: Transform) {
        self.transform = tr;
        self.transform_inverse = None;
    }

    fn transform_push(&mut self, tr: Transform) {
        self.transform = self.transform * tr;
        self.transform_inverse = None;
    }

    fn transform_get(&self) -> Transform {
        self.transform
    }

    fn transform_inverse_get(&mut self) -> Result<Transform, PSError> {
        if let Some(tr) = self.transform_inverse {
            return Ok(tr);
        }
        match self.transform.invert() {
            None => Err(PSError::NotInvertable),
            Some(tr) => {
                self.transform_inverse = Some(tr);
                Ok(tr)
            }
        }
    }
}

struct PSState {
    rasterizer: Box<dyn Rasterizer>,
    image: ImageOwned<LinColor>,
    transform_default: Transform,
    stack: Vec<PSValue>,
    dict_stack: Vec<PSDict>,
    graphic_stack: Vec<PSGraphicState>,
}

impl PSState {
    fn new() -> Self {
        Self {
            // rasterizer: Box::new(ActiveEdgeRasterizer::default()),
            rasterizer: Box::new(SignedDifferenceRasterizer::default()),
            image: ImageOwned::new_default(Size {
                width: 0,
                height: 0,
            }),
            transform_default: Transform::identity(),
            stack: Vec::new(),
            dict_stack: vec![create_global_dict(), PSDict::new()],
            graphic_stack: vec![],
        }
    }

    fn pop(&mut self) -> Result<PSValue, PSError> {
        self.stack.pop().ok_or(PSError::StackUnderflow)
    }

    fn pop_point(&mut self) -> Result<Point, PSError> {
        let y = self.pop()?.try_into_number()?;
        let x = self.pop()?.try_into_number()?;
        Ok(Point::new(x, y))
    }

    fn push(&mut self, value: impl Into<PSValue>) {
        self.stack.push(value.into())
    }

    fn load(&mut self, key: &PSSymbol) -> Result<PSValue, PSError> {
        for dict in self.dict_stack.iter().rev() {
            if let Some(val) = dict.get(key) {
                return Ok(val);
            }
        }
        Err(PSError::NotDefined(key.clone()))
    }

    fn store(&mut self, key: PSSymbol, val: impl Into<PSValue>) {
        if let Some(dict) = self.dict_stack.last_mut() {
            dict.insert(key, val.into());
        }
    }

    fn graphic(&mut self) -> &mut PSGraphicState {
        if self.graphic_stack.is_empty() {
            self.graphic_stack
                .push(PSGraphicState::new(self.transform_default));
        }
        let last_index = self.graphic_stack.len() - 1;
        &mut self.graphic_stack[last_index]
    }

    fn graphic_save(&mut self) {
        let state = self.graphic().clone();
        self.graphic_stack.push(state);
    }

    fn graphic_restore(&mut self) {
        self.graphic_stack.pop();
    }

    fn eval(&mut self, val: PSValue) -> Result<(), PSError> {
        use PSValue::*;
        match val {
            Symbol(symbol) => {
                let value = self.load(&symbol)?;
                let not_proc = value.try_run_opt(self)?;
                self.stack.extend(not_proc);
            }
            Comment(comment) => {
                if let Some(bbox_str) = comment.strip_prefix("%BoundingBox: ") {
                    let mut bbox_iter = bbox_str.split_whitespace();
                    if let Some(bbox) = (|| {
                        let x = bbox_iter.next()?.parse().ok()?;
                        let y = bbox_iter.next()?.parse().ok()?;
                        let width: Scalar = bbox_iter.next()?.parse().ok()?;
                        let height: Scalar = bbox_iter.next()?.parse().ok()?;
                        Some(BBox::new((x, y), (x + width, y + height)))
                    })() {
                        self.image = ImageOwned::new_default(Size {
                            width: bbox.width() as usize,
                            height: bbox.height() as usize,
                        });
                        self.transform_default = Transform::new_scale(1.0, -1.0)
                            .translate(-bbox.x(), -bbox.y() - bbox.height());
                    }
                }
            }
            Fn(fun) => fun.run(self)?,
            _ => self.stack.push(val),
        }
        Ok(())
    }
}

// Create default global dict
//
// Reference: Blue Book - Operator Summary
fn create_global_dict() -> PSDict {
    fn bind_fn(
        dict: &mut HashMap<PSSymbol, PSValue>,
        name: &'static str,
        def: impl Fn(&mut PSState) -> Result<(), PSError> + 'static,
    ) {
        let fun = PSFnBuildIn::new(name, def);
        dict.insert(PSSymbol(name.to_string()), PSValue::Fn(Rc::new(fun)));
    }

    let mut global_dict: HashMap<PSSymbol, PSValue> = HashMap::new();
    let dict = &mut global_dict;

    bind_fn(dict, "copy", |state| match state.pop()? {
        PSValue::Number(count) => {
            let count = count as usize;
            let range = state.stack.len() - count..state.stack.len();
            for index in range {
                let val = state.stack[index].clone();
                state.push(val);
            }
            Ok(())
        }
        PSValue::Dict(dst) => {
            let src = state.pop()?.try_into_dict()?;
            for (key, value) in src.borrow().iter() {
                dst.insert(key.clone(), value.clone());
            }
            Ok(())
        }
        val => Err(PSError::InvalidValue(val)),
    });

    // stack
    bind_fn(dict, "pop", |state| {
        state.pop()?;
        Ok(())
    });
    bind_fn(dict, "clear", |state| {
        state.stack.clear();
        Ok(())
    });
    bind_fn(dict, "exch", |state| {
        let a = state.pop()?;
        let b = state.pop()?;
        state.push(a);
        state.push(b);
        Ok(())
    });
    bind_fn(dict, "dup", |state| {
        let value = state.stack.last().ok_or(PSError::StackUnderflow)?.clone();
        state.stack.push(value);
        Ok(())
    });
    bind_fn(dict, "index", |state| {
        let index = state.pop()?.try_into_number()? as usize;
        let value = state
            .stack
            .get(state.stack.len() - index - 1)
            .ok_or_else(|| PSError::StackUnderflow)?
            .clone();
        state.stack.push(value);
        Ok(())
    });
    bind_fn(dict, "roll", |state| {
        let roll = state.pop()?.try_into_number()?;
        let size = state.pop()?.try_into_number()? as usize;
        let stack_len = state.stack.len();
        if stack_len < size {
            return Err(PSError::StackUnderflow);
        }
        let slice = &mut state.stack[stack_len - size..];
        if roll > 0.0 {
            let roll = roll as usize % size;
            slice.rotate_right(roll);
        } else {
            let roll = -roll as usize % size;
            slice.rotate_left(roll);
        }
        Ok(())
    });

    // graphics
    bind_fn(dict, "gsave", |state| {
        state.graphic_save();
        Ok(())
    });
    bind_fn(dict, "grestore", |state| {
        state.graphic_restore();
        Ok(())
    });
    bind_fn(dict, "newpath", |state| {
        state.graphic().path = Path::builder();
        Ok(())
    });
    bind_fn(dict, "clippath", |state| {
        state.graphic().clip_path = state.graphic().path.build();
        Ok(())
    });
    bind_fn(dict, "moveto", |state| {
        let point = state.pop_point()?;
        state.graphic().path.move_to(point);
        Ok(())
    });
    bind_fn(dict, "lineto", |state| {
        let point = state.pop_point()?;
        state.graphic().path.line_to(point);
        Ok(())
    });
    bind_fn(dict, "curveto", |state| {
        let p3 = state.pop_point()?;
        let p2 = state.pop_point()?;
        let p1 = state.pop_point()?;
        state.graphic().path.cubic_to(p1, p2, p3);
        Ok(())
    });
    bind_fn(dict, "closepath", |state| {
        state.graphic().path.close();
        Ok(())
    });
    bind_fn(dict, "currentpoint", |state| {
        let point = state.graphic().path.position();
        state.push(point.x());
        state.push(point.y());
        Ok(())
    });
    bind_fn(dict, "fill", |state| {
        let gstate = state.graphic();
        let transform = gstate.transform_get();
        let color = gstate.color;
        let path = gstate.path.build();
        path.fill(
            &state.rasterizer,
            transform,
            FillRule::NonZero,
            color,
            state.image.as_mut(),
        );
        Ok(())
    });
    bind_fn(dict, "stroke", |state| {
        let gstate = state.graphic();
        let transform = gstate.transform_get();
        let color = gstate.color;
        let path = gstate.path.build().stroke(gstate.stroke_style);
        path.fill(
            &state.rasterizer,
            transform,
            FillRule::NonZero,
            color,
            state.image.as_mut(),
        );
        Ok(())
    });
    bind_fn(dict, "setrgbcolor", |state| {
        let b = srgb_to_linear(state.pop()?.try_into_number()? as f32);
        let g = srgb_to_linear(state.pop()?.try_into_number()? as f32);
        let r = srgb_to_linear(state.pop()?.try_into_number()? as f32);
        state.graphic().color = LinColor::new(r, g, b, 1.0);
        Ok(())
    });
    bind_fn(dict, "setgray", |state| {
        let v = state.pop()?.try_into_number()? as f32;
        state.graphic().color = LinColor::new(v, v, v, 1.0);
        Ok(())
    });
    bind_fn(dict, "currentflat", |state| {
        state.push(1.0);
        Ok(())
    });
    bind_fn(dict, "setflat", |state| {
        let _flatness = state.pop()?.try_into_number()?;
        Ok(())
    });
    bind_fn(dict, "setlinejoin", |state| {
        let id = state.pop()?.try_into_number()? as usize;
        let line_join = match id {
            0 => LineJoin::Miter(state.graphic().miter_limit),
            1 => LineJoin::Round,
            2 => LineJoin::Bevel,
            _ => return Err(PSError::InvalidValue((id as Scalar).into())),
        };
        state.graphic().stroke_style.line_join = line_join;
        Ok(())
    });
    bind_fn(dict, "setlinecap", |state| {
        let id = state.pop()?.try_into_number()? as usize;
        let line_cap = match id {
            0 => LineCap::Butt,
            1 => LineCap::Round,
            2 => LineCap::Square,
            _ => return Err(PSError::InvalidValue((id as Scalar).into())),
        };
        state.graphic().stroke_style.line_cap = line_cap;
        Ok(())
    });
    bind_fn(dict, "setmiterlimit", |state| {
        let miter_limit = state.pop()?.try_into_number()?;
        state.graphic().miter_limit = miter_limit;
        Ok(())
    });
    bind_fn(dict, "setlinewidth", |state| {
        let width = state.pop()?.try_into_number()?;
        state.graphic().stroke_style.width = width;
        Ok(())
    });

    // transform
    let try_pop_matrix = Rc::new(
        |state: &mut PSState| -> Result<Option<Transform>, PSError> {
            match state.pop()? {
                PSValue::Matrix(transform) => Ok(Some(transform)),
                value => {
                    state.push(value);
                    Ok(None)
                }
            }
        },
    );
    bind_fn(dict, "matrix", |state| {
        state.push(Transform::identity());
        Ok(())
    });
    bind_fn(dict, "currentmatrix", |state| {
        let _ = state.pop()?.try_into_matrix()?;
        let ctm = state.graphic().transform_get();
        state.push(ctm);
        Ok(())
    });
    bind_fn(dict, "setmatrix", |state| {
        let ctm = state.pop()?.try_into_matrix()?;
        state.graphic().transform_set(ctm);
        Ok(())
    });
    bind_fn(dict, "concat", |state| {
        let tr = state.pop()?.try_into_matrix()?;
        state.graphic().transform_push(tr);
        Ok(())
    });
    bind_fn(dict, "concatmatrix", |state| {
        let _m = state.pop()?.try_into_matrix()?;
        let r = state.pop()?.try_into_matrix()?;
        let l = state.pop()?.try_into_matrix()?;
        state.push(l * r);
        Ok(())
    });
    bind_fn(dict, "invertmatrix", |state| {
        let _dst = state.pop()?.try_into_matrix()?;
        let src = state.pop()?.try_into_matrix()?;
        state.push(src.invert().ok_or(PSError::NotInvertable)?);
        Ok(())
    });
    bind_fn(dict, "transform", {
        let try_pop_matrix = try_pop_matrix.clone();
        move |state| {
            let tr = try_pop_matrix(state)?.unwrap_or(state.graphic().transform_get());
            let point = state.pop_point()?;
            let result = tr.apply(point);
            state.push(result.x());
            state.push(result.y());
            Ok(())
        }
    });
    bind_fn(dict, "itransform", {
        let try_pop_matrix = try_pop_matrix.clone();
        move |state| {
            let tr = match try_pop_matrix(state)? {
                Some(tr) => tr.invert().ok_or(PSError::NotInvertable)?,
                None => state.graphic().transform_inverse_get()?,
            };
            let point = state.pop_point()?;
            let result = tr.apply(point);
            state.push(result.x());
            state.push(result.y());
            Ok(())
        }
    });
    bind_fn(dict, "translate", {
        let try_pop_matrix = try_pop_matrix.clone();
        move |state| {
            let matrix = try_pop_matrix(state)?;
            let point = state.pop_point()?;
            let translate = Transform::new_translate(point.x(), point.y());
            match matrix {
                Some(matrix) => state.push(translate * matrix),
                None => state.graphic().transform_push(translate),
            }
            Ok(())
        }
    });
    bind_fn(dict, "scale", {
        let try_pop_matrix = try_pop_matrix.clone();
        move |state| {
            let matrix = try_pop_matrix(state)?;
            let point = state.pop_point()?;
            let scale = Transform::new_scale(point.x(), point.y());
            match matrix {
                Some(matrix) => state.push(scale * matrix),
                None => state.graphic().transform_push(scale),
            }
            Ok(())
        }
    });
    bind_fn(dict, "rotate", {
        let try_pop_matrix = try_pop_matrix.clone();
        move |state| {
            let matrix = try_pop_matrix(state)?;
            let angle = state.pop()?.try_into_number()?;
            let rotate = Transform::new_rotate(angle);
            match matrix {
                Some(matrix) => state.push(rotate * matrix),
                None => state.graphic().transform_push(rotate),
            }
            Ok(())
        }
    });

    // def
    bind_fn(dict, "def", |state| {
        let value = match state.pop()? {
            PSValue::Block(block) => PSValue::Fn(Rc::new(PSProc(block))),
            value => value,
        };
        let name = state.pop()?.try_into_symbol()?;
        state.store(name, value);
        Ok(())
    });
    bind_fn(dict, "begin", |state| {
        let dict = state.pop()?.try_into_dict()?;
        state.dict_stack.push(dict);
        Ok(())
    });
    bind_fn(dict, "end", |state| {
        let _ = state.dict_stack.pop();
        Ok(())
    });
    bind_fn(dict, "bind", |state| {
        let block = state.pop()?.try_into_block()?;
        // should this be recursive?
        let resolved = block
            .into_iter()
            .map(|val| match val {
                PSValue::Symbol(ref sym) => match state.load(sym) {
                    Ok(op @ PSValue::Fn(_)) => op,
                    _ => val,
                },
                _ => val,
            })
            .collect();
        state.push(PSValue::Block(resolved));
        Ok(())
    });

    // dict
    bind_fn(dict, "dict", |state| {
        let _size = state.pop()?;
        state.push(PSValue::Dict(PSDict::new()));
        Ok(())
    });
    bind_fn(dict, "load", |state| {
        let key = state.pop()?.try_into_symbol()?;
        let val = state.load(&key)?;
        state.push(val);
        Ok(())
    });
    bind_fn(dict, "get", |state| {
        let key = state.pop()?.try_into_symbol()?;
        let dict = state.pop()?.try_into_dict()?;
        let val = dict.get(&key).ok_or_else(|| PSError::NotDefined(key))?;
        state.push(val);
        Ok(())
    });
    bind_fn(dict, "where", |state| {
        let key = state.pop()?.try_into_symbol()?;
        for dict in state.dict_stack.iter().rev() {
            if dict.contains_key(&key) {
                let dict = dict.clone();
                state.push(dict);
                state.push(true);
                return Ok(());
            }
        }
        state.push(false);
        Ok(())
    });

    // bool
    dict.insert(PSSymbol::new("true".to_string()), PSValue::Bool(true));
    dict.insert(PSSymbol::new("false".to_string()), PSValue::Bool(false));
    bind_fn(dict, "eq", |state| {
        let a = state.pop()?;
        let b = state.pop()?;
        state.push(a == b);
        Ok(())
    });
    bind_fn(dict, "ne", |state| {
        let a = state.pop()?;
        let b = state.pop()?;
        state.push(a != b);
        Ok(())
    });
    let cmp = Rc::new(|state: &mut PSState| {
        let a = state.pop()?;
        let b = state.pop()?;
        use PSValue::*;
        match (a, b) {
            (String(a), String(b)) => Ok(b.cmp(&a)),
            (Number(a), Number(b)) => b
                .partial_cmp(&a)
                .ok_or_else(move || PSError::NotComparable(b.into(), a.into())),
            (a, b) => Err(PSError::NotComparable(b, a)),
        }
    });
    bind_fn(dict, "ge", {
        let cmp = cmp.clone();
        move |state| {
            let ord = cmp(state)?;
            state.push(ord == Ordering::Equal || ord == Ordering::Greater);
            Ok(())
        }
    });
    bind_fn(dict, "gt", {
        let cmp = cmp.clone();
        move |state| {
            let ord = cmp(state)?;
            state.push(ord == Ordering::Greater);
            Ok(())
        }
    });
    bind_fn(dict, "le", {
        let cmp = cmp.clone();
        move |state| {
            let ord = cmp(state)?;
            state.push(ord == Ordering::Equal || ord == Ordering::Less);
            Ok(())
        }
    });
    bind_fn(dict, "lt", {
        let cmp = cmp.clone();
        move |state| {
            let ord = cmp(state)?;
            state.push(ord == Ordering::Less);
            Ok(())
        }
    });

    // cond
    bind_fn(dict, "if", |state| {
        let succ = state.pop()?;
        let cond = state.pop()?.try_into_bool()?;
        if cond {
            succ.try_run(state)?;
        }
        Ok(())
    });
    bind_fn(dict, "ifelse", |state| {
        let fail = state.pop()?;
        let succ = state.pop()?;
        let cond = state.pop()?.try_into_bool()?;
        if cond {
            succ.try_run(state)?;
        } else {
            fail.try_run(state)?;
        }
        Ok(())
    });

    // math
    bind_fn(dict, "add", |state| {
        let a = state.pop()?.try_into_number()?;
        let b = state.pop()?.try_into_number()?;
        state.push(a + b);
        Ok(())
    });
    bind_fn(dict, "div", |state| {
        let a = state.pop()?.try_into_number()?;
        let b = state.pop()?.try_into_number()?;
        state.push(b / a);
        Ok(())
    });
    bind_fn(dict, "sub", |state| {
        let a = state.pop()?.try_into_number()?;
        let b = state.pop()?.try_into_number()?;
        state.push(b - a);
        Ok(())
    });
    bind_fn(dict, "neg", |state| {
        let val = state.pop()?.try_into_number()?;
        state.push(-val);
        Ok(())
    });
    bind_fn(dict, "round", |state| {
        let val = state.pop()?.try_into_number()?;
        state.push(val.round());
        Ok(())
    });

    bind_fn(dict, "pstack", |state| {
        eprintln!("STACK:");
        for val in state.stack.iter().rev() {
            eprintln!("  {:?}", val);
        }
        Ok(())
    });

    // dummy
    bind_fn(dict, "save", |state| {
        eprintln!("[not supported] save");
        state.push(PSValue::Comment("Dummy VM snapshot".to_string()));
        Ok(())
    });
    bind_fn(dict, "restore", |state| {
        eprintln!("[not supported] restore");
        state.pop()?;
        Ok(())
    });
    bind_fn(dict, "setdash", |state| {
        eprintln!("[not supported] setdash");
        let _offset = state.pop()?.try_into_number()?;
        let _array = state.pop()?.try_into_array()?;
        Ok(())
    });
    bind_fn(dict, "showpage", |_state| Ok(()));
    bind_fn(dict, "ashow", |state| {
        let _a = state.pop_point()?;
        let _text = state.pop()?.try_into_string()?;
        Ok(())
    });

    bind_fn(dict, "stop", |_state| Err(PSError::InputEmpty));

    PSDict(Rc::new(RefCell::new(global_dict)))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PSSymbol(String);

impl PSSymbol {
    fn new(string: String) -> Self {
        Self(string)
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, PartialEq)]
struct PSDict(Rc<RefCell<HashMap<PSSymbol, PSValue>>>);

impl PSDict {
    fn new() -> Self {
        Self(Rc::new(RefCell::new(HashMap::new())))
    }

    fn get(&self, key: &PSSymbol) -> Option<PSValue> {
        self.borrow().get(key).cloned()
    }

    fn contains_key(&self, key: &PSSymbol) -> bool {
        self.borrow().contains_key(key)
    }

    fn insert(&self, key: PSSymbol, val: impl Into<PSValue>) -> Option<PSValue> {
        self.borrow_mut().insert(key, val.into())
    }

    fn borrow(&self) -> cell::Ref<'_, HashMap<PSSymbol, PSValue>> {
        self.0.borrow()
    }

    fn borrow_mut(&self) -> cell::RefMut<'_, HashMap<PSSymbol, PSValue>> {
        self.0.borrow_mut()
    }
}

impl fmt::Debug for PSDict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.borrow().fmt(f)
    }
}

trait PSFn: fmt::Debug {
    fn run(&self, state: &mut PSState) -> Result<(), PSError>;
}

struct PSFnBuildIn<F> {
    name: &'static str,
    fun: F,
}

impl<F> PSFnBuildIn<F> {
    fn new(name: &'static str, fun: F) -> Self {
        Self { name, fun }
    }
}

impl<F> PSFn for PSFnBuildIn<F>
where
    F: Fn(&mut PSState) -> Result<(), PSError>,
{
    fn run(&self, state: &mut PSState) -> Result<(), PSError> {
        (self.fun)(state)
    }
}

impl<F> fmt::Debug for PSFnBuildIn<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name.fmt(f)
    }
}

struct PSProc(Vec<PSValue>);

impl PSFn for PSProc {
    fn run(&self, state: &mut PSState) -> Result<(), PSError> {
        for val in self.0.iter() {
            state.eval(val.clone())?;
        }
        Ok(())
    }
}

impl fmt::Debug for PSProc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

#[derive(Clone)]
enum PSValue {
    Bool(bool),
    Number(Scalar),
    Symbol(PSSymbol),
    Quote(PSSymbol),
    String(String),
    Comment(String),
    Block(Vec<PSValue>),
    Array(Vec<PSValue>),
    Matrix(Transform),
    Dict(PSDict),
    Fn(Rc<dyn PSFn>),
}

impl PSValue {
    fn try_into_symbol(self) -> Result<PSSymbol, PSError> {
        match self {
            PSValue::Quote(symbol) => Ok(symbol),
            _ => Err(PSError::ExpectedQuote),
        }
    }

    fn try_into_number(self) -> Result<Scalar, PSError> {
        match self {
            PSValue::Number(value) => Ok(value),
            _ => Err(PSError::ExpectedNumber),
        }
    }

    fn try_into_dict(self) -> Result<PSDict, PSError> {
        match self {
            PSValue::Dict(dict) => Ok(dict),
            _ => Err(PSError::ExpectedDict),
        }
    }

    fn try_into_array(self) -> Result<Vec<PSValue>, PSError> {
        match self {
            PSValue::Array(array) => Ok(array),
            _ => Err(PSError::ExpectedArray),
        }
    }

    fn try_into_matrix(self) -> Result<Transform, PSError> {
        match self {
            PSValue::Matrix(tr) => Ok(tr),
            _ => Err(PSError::ExpectedMatrix),
        }
    }

    fn try_into_block(self) -> Result<Vec<PSValue>, PSError> {
        match self {
            PSValue::Block(block) => Ok(block),
            _ => Err(PSError::ExpectedBlock),
        }
    }

    fn try_into_bool(self) -> Result<bool, PSError> {
        match self {
            PSValue::Bool(val) => Ok(val),
            _ => Err(PSError::ExpectedBool),
        }
    }

    fn try_into_string(self) -> Result<String, PSError> {
        match self {
            PSValue::String(string) => Ok(string),
            _ => Err(PSError::ExpectedBool),
        }
    }

    fn try_run_opt(self, state: &mut PSState) -> Result<Option<PSValue>, PSError> {
        match self {
            PSValue::Fn(fun) => fun.run(state)?,
            PSValue::Block(block) => {
                // useful for conditionals
                for val in block {
                    state.eval(val)?;
                }
            }
            val => return Ok(Some(val)),
        }
        Ok(None)
    }

    fn try_run(self, state: &mut PSState) -> Result<(), PSError> {
        match self.try_run_opt(state)? {
            None => Ok(()),
            Some(_) => Err(PSError::ExpectedProc),
        }
    }
}

impl PartialEq<PSValue> for PSValue {
    fn eq(&self, other: &PSValue) -> bool {
        use PSValue::*;
        match (self, other) {
            (Bool(a), Bool(b)) => a.eq(b),
            (Number(a), Number(b)) => a.eq(b),
            (Symbol(a), Symbol(b)) => a.eq(b),
            (Quote(a), Quote(b)) => a.eq(b),
            (String(a), String(b)) => a.eq(b),
            (Comment(a), Comment(b)) => a.eq(b),
            (Block(a), Block(b)) => a.eq(b),
            (Array(a), Array(b)) => a.eq(b),
            (Dict(a), Dict(b)) => a.eq(b),
            _ => false,
        }
    }
}

impl From<Scalar> for PSValue {
    fn from(value: Scalar) -> Self {
        PSValue::Number(value)
    }
}

impl From<bool> for PSValue {
    fn from(value: bool) -> Self {
        PSValue::Bool(value)
    }
}

impl From<PSDict> for PSValue {
    fn from(dict: PSDict) -> Self {
        PSValue::Dict(dict)
    }
}

impl From<Transform> for PSValue {
    fn from(tr: Transform) -> Self {
        PSValue::Matrix(tr)
    }
}

impl fmt::Debug for PSValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use PSValue::*;
        match self {
            Bool(val) => f.debug_tuple("Bool").field(val).finish(),
            Number(num) => f.debug_tuple("Number").field(num).finish(),
            Symbol(sym) => f.debug_tuple("Symbol").field(&sym.as_str()).finish(),
            Quote(sym) => f.debug_tuple("Quote").field(&sym.as_str()).finish(),
            String(string) => f.debug_tuple("String").field(string).finish(),
            Comment(string) => f.debug_tuple("Comment").field(string).finish(),
            Block(block) => f.debug_tuple("Block").field(&block).finish(),
            Array(array) => f.debug_tuple("Array").field(&array).finish(),
            Dict(dict) => f.debug_tuple("Dict").field(&dict).finish(),
            Matrix(tr) => f.debug_tuple("Matrix").field(&tr).finish(),
            Fn(fun) => f.debug_tuple("Fn").field(&fun).finish(),
        }
    }
}

struct PSParser<I> {
    input: I,
    buf: Vec<u8>,
}

impl<I: Read> PSParser<I> {
    fn new(input: I) -> Self {
        Self {
            input,
            buf: Default::default(),
        }
    }

    fn parse(&mut self) -> Result<PSValue, PSError> {
        let byte = loop {
            let byte = self.read_byte()?;
            if !matches!(byte, b' ' | b'\n' | b'\r' | b'\t' | b'\x00' | b'\x0c') {
                break byte;
            }
        };

        match byte {
            b'/' => {
                let symbol = PSSymbol::new(self.read_symbol()?);
                Ok(PSValue::Quote(symbol))
            }
            b'%' => Ok(PSValue::Comment(self.read_while(|b| b != b'\n')?)),
            b'-' | b'+' | b'0'..=b'9' | b'.' => {
                self.push_byte(byte);
                let number = self.read_number()?;
                Ok(PSValue::Number(number))
            }
            b'_' | b'a'..=b'z' | b'A'..=b'Z' => {
                self.push_byte(byte);
                let symbol = PSSymbol::new(self.read_symbol()?);
                Ok(PSValue::Symbol(symbol))
            }
            b'(' => {
                let string = self.read_while(|byte| byte != b')')?;
                self.read_byte()?;
                Ok(PSValue::String(string))
            }
            b'{' => {
                let mut block = Vec::new();
                loop {
                    match self.parse() {
                        Ok(cmd) => block.push(cmd),
                        Err(PSError::UnclosedBlock) => break,
                        Err(error) => return Err(error),
                    }
                }
                Ok(PSValue::Block(block))
            }
            b'}' => Err(PSError::UnclosedBlock),
            b'[' => {
                let mut array = Vec::new();
                loop {
                    match self.parse() {
                        Ok(cmd) => array.push(cmd),
                        Err(PSError::UnclosedArray) => break,
                        Err(error) => return Err(error),
                    }
                }
                Ok(PSValue::Array(array))
            }
            b']' => Err(PSError::UnclosedArray),
            _ => Err(PSError::InputUnexpected(byte)),
        }
    }

    fn read_byte(&mut self) -> Result<u8, PSError> {
        match self.buf.pop() {
            None => {
                let mut byte = [0; 1];
                if let Err(error) = self.input.read_exact(&mut byte[..]) {
                    if error.kind() == std::io::ErrorKind::UnexpectedEof {
                        return Err(PSError::InputEmpty);
                    }
                    return Err(error.into());
                }
                Ok(byte[0])
            }
            Some(byte) => Ok(byte),
        }
    }

    fn push_byte(&mut self, byte: u8) {
        self.buf.push(byte)
    }

    fn read_while(&mut self, mut pred: impl FnMut(u8) -> bool) -> Result<String, PSError> {
        let mut result = Vec::new();
        loop {
            match self.read_byte() {
                Ok(byte) => {
                    if pred(byte) {
                        result.push(byte);
                    } else {
                        self.push_byte(byte);
                        break;
                    }
                }
                Err(PSError::InputEmpty) => break,
                Err(error) => return Err(error),
            }
        }
        Ok(String::from_utf8(result)?)
    }

    fn read_number(&mut self) -> Result<Scalar, PSError> {
        let number =
            self.read_while(|byte| matches!(byte, b'0'..=b'9' | b'.' | b'+' | b'-' | b'e' | b'E'))?;
        Ok(Scalar::from_str(&number)?)
    }

    fn read_symbol(&mut self) -> Result<String, PSError> {
        let symbol = self.read_while(|byte| {
            !matches!(
                byte,
                b' ' | b'\n' | b'\r' | b'\t' | b'\x00' | b'\x0c' | b'[' | b']' | b'{' | b'}' | b'/'
            )
        })?;
        Ok(symbol)
    }
}

impl<I: Read> Iterator for PSParser<I> {
    type Item = Result<PSValue, PSError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.parse() {
            Err(PSError::InputEmpty) => None,
            item => Some(item),
        }
    }
}

#[derive(Debug)]
enum PSError {
    IOError(std::io::Error),
    Float(ParseFloatError),
    Utf8(FromUtf8Error),
    InputUnexpected(u8),
    InputEmpty,
    UnclosedArray,
    UnclosedBlock,
    ExpectedQuote,
    ExpectedNumber,
    ExpectedDict,
    ExpectedArray,
    ExpectedMatrix,
    ExpectedBlock,
    ExpectedProc,
    ExpectedBool,
    NotDefined(PSSymbol),
    NotComparable(PSValue, PSValue),
    NotInvertable,
    InvalidValue(PSValue),
    StackUnderflow,
}

impl From<std::io::Error> for PSError {
    fn from(error: std::io::Error) -> Self {
        Self::IOError(error)
    }
}

impl From<FromUtf8Error> for PSError {
    fn from(error: FromUtf8Error) -> Self {
        Self::Utf8(error)
    }
}

impl From<ParseFloatError> for PSError {
    fn from(error: ParseFloatError) -> Self {
        Self::Float(error)
    }
}

fn main() -> Result<(), PSError> {
    use std::{
        fs::File,
        io::{BufReader, BufWriter},
    };

    // let file = BufReader::new(File::open("/mnt/data/downloads/tiger.eps")?);
    let file = BufReader::new(File::open("/mnt/data/downloads/tiger.eps")?);
    let parser = PSParser::new(file);
    let mut state = PSState::new();
    for val in parser {
        let val = val?;
        // println!("{:?}", val);
        if let Err(err) = state.eval(val) {
            if let PSError::InputEmpty = err {
                break;
            }
            eprintln!("ERROR: {:?}", err);
            for val in PSParser::new(std::io::stdin()) {
                if let Err(err) = state.eval(val?) {
                    eprintln!("ERROR: {:?}", err);
                }
            }
            return Ok(());
        }
    }

    let mut image = BufWriter::new(File::create("/tmp/eps.bmp")?);
    state.image.write_bmp(&mut image)?;

    Ok(())
}
