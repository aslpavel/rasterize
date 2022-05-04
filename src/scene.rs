use crate::{
    utils::clamp, BBox, Color, FillRule, Image, ImageMut, ImageOwned, LinColor, Paint, Path, Point,
    Rasterizer, Scalar, Shape, Size, StrokeStyle, Transform, Units,
};
use std::{cmp, fmt, sync::Arc};

#[derive(Debug)]
pub enum SceneInner {
    Fill {
        path: Arc<Path>,
        paint: Arc<dyn Paint>,
        fill_rule: FillRule,
    },
    Stroke {
        path: Arc<Path>,
        paint: Arc<dyn Paint>,
        style: StrokeStyle,
    },
    Group {
        children: Vec<Scene>,
    },
    Transform {
        child: Scene,
        tr: Transform,
    },
    Opacity {
        child: Scene,
        opacity: Scalar,
    },
    Clip {
        child: Scene,
        clip: Arc<Path>,
        units: Units,
        fill_rule: FillRule,
    },
}

#[derive(Clone)]
pub struct Scene {
    inner: Arc<SceneInner>,
}

impl AsRef<SceneInner> for Scene {
    fn as_ref(&self) -> &SceneInner {
        self.inner.as_ref()
    }
}

impl From<SceneInner> for Scene {
    fn from(inner: SceneInner) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl Scene {
    pub fn fill(path: Arc<Path>, paint: Arc<dyn Paint>, fill_rule: FillRule) -> Self {
        SceneInner::Fill {
            path,
            paint,
            fill_rule,
        }
        .into()
    }

    pub fn stroke(path: Arc<Path>, paint: Arc<dyn Paint>, style: StrokeStyle) -> Self {
        SceneInner::Stroke { path, paint, style }.into()
    }

    pub fn group(children: Vec<Scene>) -> Self {
        match children.as_slice() {
            [child] => child.clone(),
            _ => SceneInner::Group { children }.into(),
        }
    }

    pub fn opacity(&self, opacity: Scalar) -> Self {
        let opacity = clamp(opacity, 0.0, 1.0);
        match self.as_ref() {
            SceneInner::Opacity {
                child,
                opacity: child_opacity,
            } => child.opacity(opacity * child_opacity),
            _ => SceneInner::Opacity {
                child: self.clone(),
                opacity,
            }
            .into(),
        }
    }

    pub fn clip(&self, clip: Arc<Path>, units: Units, fill_rule: FillRule) -> Self {
        SceneInner::Clip {
            child: self.clone(),
            clip,
            units,
            fill_rule,
        }
        .into()
    }

    pub fn transform(&self, tr: Transform) -> Self {
        match self.as_ref() {
            SceneInner::Transform {
                child,
                tr: child_tr,
            } => child.transform(tr * *child_tr),
            _ => SceneInner::Transform {
                child: self.clone(),
                tr,
            }
            .into(),
        }
    }

    pub fn bbox(&self, tr: Transform) -> Option<BBox> {
        use SceneInner::*;
        match self.as_ref() {
            Fill { path, .. } => path.bbox(tr),
            Stroke { path, .. } => path.bbox(tr),
            Group { children } => children.iter().fold(None, |bbox, child| match bbox {
                Some(bbox) => Some(bbox.union_opt(child.bbox(tr))),
                None => child.bbox(tr),
            }),
            Transform {
                child,
                tr: child_tr,
            } => child.bbox(tr * *child_tr),
            Opacity { child, .. } => child.bbox(tr),
            Clip {
                child, clip, units, ..
            } => {
                let clip_tr = match units {
                    Units::UserSpaceOnUse => tr,
                    Units::BoundingBox => {
                        let bbox = child.bbox(Default::default())?;
                        tr * bbox.unit_transform()
                    }
                };
                let child_bbox = child.bbox(tr)?;
                let clip_bbox = clip.bbox(clip_tr)?;
                child_bbox.intersect(clip_bbox)
            }
        }
    }

    pub fn render(
        &self,
        rasterizer: &dyn Rasterizer,
        tr: Transform,
        view: Option<BBox>,
        bg: Option<LinColor>,
    ) -> Layer<LinColor> {
        let view = match view.or_else(|| self.bbox(tr)) {
            None => return Layer::empty(),
            Some(view) => view,
        };
        let mut layer = Layer::new(view, bg);
        self.render_rec(rasterizer, &mut layer, tr, None);
        layer
    }

    fn render_rec(
        &self,
        rasterizer: &dyn Rasterizer,
        layer: &mut Layer<LinColor>,
        tr: Transform,
        quick_opacity: Option<Scalar>,
    ) {
        use SceneInner::*;
        match self.as_ref() {
            Fill {
                path,
                paint,
                fill_rule,
            } => {
                let align =
                    crate::Transform::new_translate(-layer.x() as Scalar, -layer.y() as Scalar);
                match quick_opacity {
                    None => {
                        path.fill(rasterizer, align * tr, *fill_rule, paint, layer);
                    }
                    Some(opacity) => {
                        let paint = OpacityPaint {
                            paint,
                            opacity: opacity as f32,
                        };
                        path.fill(rasterizer, align * tr, *fill_rule, paint, layer);
                    }
                }
            }
            Stroke { path, paint, style } => {
                let path = path.stroke(*style);
                let align =
                    crate::Transform::new_translate(-layer.x() as Scalar, -layer.y() as Scalar);
                match quick_opacity {
                    None => {
                        path.fill(rasterizer, align * tr, FillRule::NonZero, paint, layer);
                    }
                    Some(opacity) => {
                        let paint = OpacityPaint {
                            paint,
                            opacity: opacity as f32,
                        };
                        path.fill(rasterizer, align * tr, FillRule::NonZero, paint, layer);
                    }
                }
            }
            Group { children } => {
                for child in children {
                    child.render_rec(rasterizer, layer, tr, quick_opacity)
                }
            }
            Transform {
                child,
                tr: child_tr,
            } => {
                child.render_rec(rasterizer, layer, tr * *child_tr, quick_opacity);
            }
            Opacity { child, opacity } => {
                match quick_opacity {
                    Some(quick_opacity) => {
                        child.render_rec(rasterizer, layer, tr, Some(opacity * quick_opacity))
                    }
                    None if child.is_quick_opacity() => {
                        child.render_rec(rasterizer, layer, tr, Some(*opacity))
                    }
                    None => {
                        let bbox = match child.bbox(tr) {
                            None => return,
                            Some(bbox) => bbox,
                        };
                        // account for anti-aliasing
                        let bbox = bbox.extend(bbox.max() + Point::new(1.0, 1.0));

                        let child_layer = child.render(rasterizer, tr, Some(bbox), None);

                        let opacity = *opacity as f32;
                        layer.compose(&child_layer, |dst, src| {
                            let src = src * opacity;
                            dst.blend_over(&src)
                        });
                    }
                }
            }
            Clip {
                child,
                clip,
                units,
                fill_rule,
            } => {
                let bbox = match self.bbox(tr) {
                    None => return,
                    Some(bbox) => bbox,
                };
                let bbox = bbox.extend(bbox.max() + Point::new(1.0, 1.0));

                // mask
                let clip_tr = match units {
                    Units::UserSpaceOnUse => tr,
                    Units::BoundingBox => {
                        let bbox = match child.bbox(Default::default()) {
                            None => return,
                            Some(bbox) => bbox,
                        };
                        tr * bbox.unit_transform()
                    }
                };
                let mut mask = Layer::new(bbox, None);
                let align =
                    crate::Transform::new_translate(-mask.x() as Scalar, -mask.y() as Scalar);
                clip.mask(rasterizer, align * clip_tr, *fill_rule, mask.as_mut());

                // child
                let mut child_layer = child.render(rasterizer, tr, Some(bbox), None);

                // compose
                child_layer.compose(&mask, |dst, src| dst * (src as f32));
                layer.compose(&child_layer, |dst, src| dst.blend_over(&src));
            }
        }
    }

    // Check if quick opacity can be used
    //
    // This function checks if all rendering nodes are non-intersecting, and as the result
    // it is safe to pre-multiply brush with opacity instead of doing offscreen rendering
    fn is_quick_opacity(&self) -> bool {
        fn is_quick_opacity_rec(scene: &Scene, boxes: &mut Vec<BBox>, tr: Transform) -> bool {
            use SceneInner::*;
            match scene.as_ref() {
                Fill { path, .. } => disjoint(boxes, path.bbox(tr)),
                Stroke { path, .. } => disjoint(boxes, path.bbox(tr)),
                Group { children } => {
                    for child in children {
                        if !is_quick_opacity_rec(child, boxes, tr) {
                            return false;
                        }
                    }
                    true
                }
                Transform {
                    child,
                    tr: child_tr,
                } => is_quick_opacity_rec(child, boxes, tr * *child_tr),
                Opacity { child, .. } => is_quick_opacity_rec(child, boxes, tr),
                Clip { child, .. } => is_quick_opacity_rec(child, boxes, tr),
            }
        }

        fn disjoint(boxes: &mut Vec<BBox>, other: Option<BBox>) -> bool {
            let other = match other {
                None => return true,
                Some(other) => other,
            };
            for bbox in boxes.iter() {
                if bbox.intersect(other).is_some() {
                    return false;
                }
            }
            boxes.push(other);
            true
        }

        let mut boxes = Vec::new();
        is_quick_opacity_rec(self, &mut boxes, Transform::identity())
    }
}

impl fmt::Debug for Scene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineNodeId(usize);

enum PipelineItem {
    Fill {
        path: Arc<Path>,
        paint: Arc<dyn Paint>,
        fill_rule: FillRule,
    },
    Group {
        children: Vec<PipelineNodeId>,
    },
    Opacity {
        child: PipelineNodeId,
        opacity: Scalar,
    },
    Clip {
        child: PipelineNodeId,
        clip: Arc<Path>,
        fill_rule: FillRule,
        units: Units,
    },
}

struct PipelineNode {
    item: PipelineItem,
    // rendering bounding box (in case of a stroke includes stroke itself)
    bbox: Option<BBox>,
    // transform used for rendering
    tr: Transform,
}

struct Pipeline {
    nodes: Vec<PipelineNode>,
}

impl Pipeline {
    pub fn build(
        scene: &Scene,
        tr: Transform,
        view: Option<BBox>,
        // rasterizer: &dyn Rasterizer,
        // bg: Option<LinColor>,
    ) -> Self {
        let mut pipeline = Self { nodes: Vec::new() };

        fn build_rec(
            pipeline: &mut Pipeline,
            scene: &Scene,
            view: Option<BBox>,
            tr: Transform,
        ) -> PipelineNodeId {
            use SceneInner::*;
            // allocate node
            let id = match scene.as_ref() {
                Fill {
                    path,
                    paint,
                    fill_rule,
                } => {
                    let item = PipelineItem::Fill {
                        path: path.clone(),
                        paint: paint.clone(),
                        fill_rule: *fill_rule,
                    };
                    pipeline.alloc(item, path.bbox(tr), tr)
                }
                Stroke { path, paint, style } => {
                    let path = path.stroke(*style);
                    let bbox = path.bbox(tr);
                    let item = PipelineItem::Fill {
                        path: Arc::new(path),
                        paint: paint.clone(),
                        fill_rule: FillRule::NonZero,
                    };
                    pipeline.alloc(item, bbox, tr)
                }
                Group { children } => {
                    let children = children
                        .iter()
                        .map(|child| build_rec(pipeline, child, view, tr))
                        .collect();
                    unimplemented!()
                }
            };
            // reduce bounding box
            if let Some(view) = view {
                let mut node = pipeline.get_mut(id);
                if let Some(bbox) = node.bbox {
                    node.bbox = view.intersect(bbox);
                }
            }
            id
        }

        build_rec(&mut pipeline, scene, view, tr);
        pipeline
    }

    fn alloc(&mut self, item: PipelineItem, bbox: Option<BBox>, tr: Transform) -> PipelineNodeId {
        let id = PipelineNodeId(self.nodes.len());
        self.nodes.push(PipelineNode { item, bbox, tr });
        id
    }

    fn get_mut(&mut self, id: PipelineNodeId) -> &mut PipelineNode {
        &mut self.nodes[id.0]
    }
}

#[derive(Debug)]
struct OpacityPaint<P> {
    paint: P,
    opacity: f32,
}

impl<P: Paint> Paint for OpacityPaint<P> {
    fn at(&self, point: Point) -> LinColor {
        self.paint.at(point) * self.opacity
    }

    fn units(&self) -> Option<Units> {
        self.paint.units()
    }

    fn transform(&self) -> Transform {
        self.paint.transform()
    }
}

/// An `OwnedImage` together with an offset
#[derive(Debug, Clone)]
pub struct Layer<C> {
    image: ImageOwned<C>,
    x: i32,
    y: i32,
}

impl<C: Default + Copy> Layer<C> {
    pub fn new(bbox: BBox, color: Option<C>) -> Self {
        let x0 = bbox.min().x().floor() as i32;
        let x1 = bbox.max().x().ceil() as i32;
        let y0 = bbox.min().y().floor() as i32;
        let y1 = bbox.max().y().ceil() as i32;
        let size = Size {
            width: (x1 - x0) as usize,
            height: (y1 - y0) as usize,
        };
        let image = match color {
            None => ImageOwned::new_default(size),
            Some(color) => ImageOwned::new_with(size, |_, _| color),
        };
        Self {
            image,
            x: x0,
            y: y0,
        }
    }

    pub fn empty() -> Self {
        Self {
            image: ImageOwned::empty(),
            x: 0,
            y: 0,
        }
    }

    pub fn x(&self) -> i32 {
        self.x
    }

    pub fn y(&self) -> i32 {
        self.y
    }

    pub fn translate(self, dx: i32, dy: i32) -> Self {
        Self {
            image: self.image,
            x: self.x + dx,
            y: self.y + dy,
        }
    }

    pub fn compose<CF, CO>(&mut self, other: &Layer<CO>, compose: CF)
    where
        CO: Default + Copy,
        CF: Fn(C, CO) -> C,
    {
        let x0 = cmp::max(self.x, other.x);
        let x1 = cmp::min(self.x + self.width() as i32, other.x + other.width() as i32);
        let y0 = cmp::max(self.y, other.y);
        let y1 = cmp::min(
            self.y + self.height() as i32,
            other.y + other.height() as i32,
        );

        let dst_x = self.x();
        let dst_y = self.y();
        let dst_shape = self.shape();
        let dst_data = self.data_mut();
        let src_shape = other.shape();
        let src_data = other.data();

        for x in x0..x1 {
            for y in y0..y1 {
                let dst_col = (x - dst_x) as usize;
                let dst_row = (y - dst_y) as usize;
                let src_col = (x - other.x()) as usize;
                let src_row = (y - other.y()) as usize;

                let dst = &mut dst_data[dst_shape.offset(dst_row, dst_col)];
                let src = src_data[src_shape.offset(src_row, src_col)];
                *dst = compose(*dst, src);
            }
        }
    }
}

impl<C> Image for Layer<C> {
    type Pixel = C;

    fn shape(&self) -> Shape {
        self.image.shape()
    }

    fn data(&self) -> &[Self::Pixel] {
        self.image.data()
    }
}

impl<C> ImageMut for Layer<C> {
    fn data_mut(&mut self) -> &mut [Self::Pixel] {
        self.image.data_mut()
    }
}
