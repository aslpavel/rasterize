use crate::{
    BBox, Color, FillRule, Image, ImageMut, ImageOwned, LinColor, Paint, Path, Point, Rasterizer,
    Scalar, Shape, Size, Transform,
};
use std::{cmp, sync::Arc};

pub enum SceneInner {
    Fill {
        path: Arc<Path>,
        paint: Arc<dyn Paint>,
        fill_rule: FillRule,
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

    pub fn group(children: Vec<Scene>) -> Self {
        match children.as_slice() {
            [child] => child.clone(),
            _ => SceneInner::Group { children }.into(),
        }
    }

    pub fn opacity(&self, opacity: Scalar) -> Self {
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
            Group { children } => children.iter().fold(None, |bbox, child| match bbox {
                Some(bbox) => Some(bbox.union_opt(child.bbox(tr))),
                None => child.bbox(tr),
            }),
            Transform {
                child,
                tr: child_tr,
            } => child.bbox(tr * *child_tr),
            Opacity { child, .. } => child.bbox(tr),
        }
    }

    pub fn render(&self, rasterizer: &dyn Rasterizer, tr: Transform, view: BBox) -> Layer {
        let mut layer = Layer::new(view);
        self.render_rec(rasterizer, &mut layer, tr, None);
        layer
    }

    fn render_rec(
        &self,
        rasterizer: &dyn Rasterizer,
        layer: &mut Layer,
        tr: Transform,
        quick_opacity: Option<Scalar>,
    ) {
        use SceneInner::*;
        match self.as_ref() {
            Fill {
                path,
                paint,
                fill_rule,
            } => match quick_opacity {
                None => {
                    path.fill(rasterizer, tr, *fill_rule, paint, layer);
                }
                Some(opacity) => {
                    let paint = OpacityPaint {
                        paint,
                        opacity: opacity as f32,
                    };
                    path.fill(rasterizer, tr, *fill_rule, paint, layer);
                }
            },
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
                        if let Some(child_bbox) = child.bbox(tr) {
                            // accout for anti-aliasing
                            let child_bbox =
                                child_bbox.extend(child_bbox.max() + Point::new(1.0, 1.0));
                            let child_layer = child.render(rasterizer, tr, child_bbox);
                            let opacity = *opacity as f32;
                            layer.compose(&child_layer, |dst, src| {
                                let src = src * opacity;
                                dst.blend_over(&src)
                            });
                        }
                    }
                }
            }
        }
    }

    // Check if quick opacity can be used
    //
    // This function checks if all rendering nodes are non-intersecting, and as the result
    // it is safe to premultiply brush with opacity instead of doing ofscreen rendering
    fn is_quick_opacity(&self) -> bool {
        fn is_quick_opacity_rec(scene: &Scene, boxes: &mut Vec<BBox>, tr: Transform) -> bool {
            use SceneInner::*;
            match scene.as_ref() {
                Fill { path, .. } => disjoint(boxes, path.bbox(tr)),
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
        is_quick_opacity_rec(self, &mut boxes, Transform::default())
    }
}

struct OpacityPaint<P> {
    paint: P,
    opacity: f32,
}

impl<P: Paint> Paint for OpacityPaint<P> {
    fn at(&self, point: Point) -> LinColor {
        self.paint.at(point) * self.opacity
    }
}

pub struct Layer {
    image: ImageOwned<LinColor>,
    x: usize,
    y: usize,
}

impl Layer {
    pub fn new(bbox: BBox) -> Self {
        let x0 = bbox.min().x().floor() as usize;
        let y0 = bbox.min().y().floor() as usize;
        let x1 = bbox.max().x().ceil() as usize;
        let y1 = bbox.max().y().ceil() as usize;
        let size = Size {
            width: x1 - x0,
            height: y1 - y0,
        };
        let image = ImageOwned::new_default(size);
        Self {
            image,
            x: x0,
            y: y0,
        }
    }

    pub fn x(&self) -> usize {
        self.x
    }

    pub fn y(&self) -> usize {
        self.y
    }

    pub fn compose<C>(&mut self, other: &Layer, compose: C)
    where
        C: Fn(LinColor, LinColor) -> LinColor,
    {
        let x0 = cmp::max(self.x(), other.x());
        let x1 = cmp::min(self.x() + self.width(), other.x() + other.width());
        let y0 = cmp::max(self.y(), other.y());
        let y1 = cmp::min(self.y() + self.height(), other.x() + other.height());
        let dst_shape = self.shape();
        let dst_x = self.x();
        let dst_y = self.y();
        let dst_data = self.data_mut();
        let src_shape = other.shape();
        let src_data = other.data();
        for x in x0..x1 {
            for y in y0..y1 {
                let dst_col = x - dst_x;
                let dst_row = y - dst_y;
                let src_col = x - other.x();
                let src_row = y - other.y();
                let dst = &mut dst_data[dst_shape.offset(dst_row, dst_col)];
                let src = src_data[src_shape.offset(src_row, src_col)];
                *dst = compose(*dst, src);
            }
        }
    }
}

impl Image for Layer {
    type Pixel = LinColor;

    fn shape(&self) -> Shape {
        self.image.shape()
    }

    fn data(&self) -> &[Self::Pixel] {
        self.image.data()
    }
}

impl ImageMut for Layer {
    fn data_mut(&mut self) -> &mut [Self::Pixel] {
        self.image.data_mut()
    }
}
