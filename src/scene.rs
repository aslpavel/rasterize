use crate::{
    utils::clamp, BBox, Color, FillRule, Image, ImageMut, ImageOwned, LinColor, Paint, Path,
    Rasterizer, Scalar, Shape, Size, StrokeStyle, Transform, Units,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{cmp, fmt, sync::Arc};

#[derive(Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(tag = "type", rename_all = "lowercase")
)]
enum SceneInner {
    Fill {
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "crate::utils::is_default")
        )]
        fill_rule: FillRule,
        #[cfg_attr(feature = "serde", serde(with = "serde_paint"))]
        paint: Arc<dyn Paint>,
        path: Arc<Path>,
    },
    Stroke {
        #[cfg_attr(feature = "serde", serde(flatten))]
        style: StrokeStyle,
        #[cfg_attr(feature = "serde", serde(with = "serde_paint"))]
        paint: Arc<dyn Paint>,
        path: Arc<Path>,
    },
    Group {
        children: Vec<Scene>,
    },
    Transform {
        #[cfg_attr(feature = "serde", serde(with = "crate::utils::serde_from_str"))]
        tr: Transform,
        child: Scene,
    },
    Opacity {
        child: Scene,
        opacity: Scalar,
    },
    Clip {
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "crate::utils::is_default")
        )]
        fill_rule: FillRule,
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "crate::utils::is_default")
        )]
        units: Units,
        clip: Arc<Path>,
        child: Scene,
    },
}

/// Represents an image that has not been rendered yet, multiple scenes
/// can be composed to construct more complex scene.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
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
    /// Fill path
    pub fn fill(path: Arc<Path>, paint: Arc<dyn Paint>, fill_rule: FillRule) -> Self {
        SceneInner::Fill {
            path,
            paint,
            fill_rule,
        }
        .into()
    }

    /// Stroke path
    pub fn stroke(path: Arc<Path>, paint: Arc<dyn Paint>, style: StrokeStyle) -> Self {
        SceneInner::Stroke { path, paint, style }.into()
    }

    /// Group multiple sub-scenes
    pub fn group(children: Vec<Scene>) -> Self {
        match children.as_slice() {
            [child] => child.clone(),
            _ => SceneInner::Group { children }.into(),
        }
    }

    /// Apply opacity to the scene
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

    /// Apply clip to the scene
    pub fn clip(&self, clip: Arc<Path>, units: Units, fill_rule: FillRule) -> Self {
        SceneInner::Clip {
            child: self.clone(),
            clip,
            units,
            fill_rule,
        }
        .into()
    }

    /// Apply transformation to the scene
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

    /// Bounding box of the scene, not including stroke width
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

    /// Render scene
    pub fn render(
        &self,
        rasterizer: &dyn Rasterizer,
        tr: Transform,
        view: Option<BBox>,
        bg: Option<LinColor>,
    ) -> Layer<LinColor> {
        let pipeline = Pipeline::build(self, tr, view);
        let root_id = match pipeline.root() {
            None => return Layer::empty(),
            Some(root_id) => root_id,
        };
        pipeline.render(rasterizer, root_id, view, bg)
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
        clip_tr: Transform,
        fill_rule: FillRule,
    },
}

struct PipelineNode {
    item: PipelineItem,
    // rendering bounding box (in case of a stroke includes stroke itself)
    bbox: BBox,
    // transform used for rendering
    tr: Transform,
}

/// Scene with enriched with additional information (such as bounding box)
/// which allows for better rendering optimizations.
struct Pipeline {
    nodes: Vec<PipelineNode>,
}

impl Pipeline {
    /// Build pipeline
    pub fn build(scene: &Scene, tr: Transform, view: Option<BBox>) -> Self {
        let mut pipeline = Self { nodes: Vec::new() };
        pipeline.build_rec(scene, view, tr);
        pipeline
    }

    fn build_rec(
        &mut self,
        scene: &Scene,
        view: Option<BBox>,
        tr: Transform,
    ) -> Option<PipelineNodeId> {
        use SceneInner::*;

        // restrict bounding box to view
        fn view_apply(view: Option<BBox>, bbox: Option<BBox>) -> Option<BBox> {
            match view {
                None => bbox,
                Some(view) => view.intersect(bbox?),
            }
        }

        match scene.as_ref() {
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
                Some(self.alloc(item, view_apply(view, path.bbox(tr))?, tr))
            }
            Stroke { path, paint, style } => {
                let path = path.stroke(*style);
                let bbox = view_apply(view, path.bbox(tr))?;
                let item = PipelineItem::Fill {
                    path: Arc::new(path),
                    paint: paint.clone(),
                    fill_rule: FillRule::NonZero,
                };
                Some(self.alloc(item, bbox, tr))
            }
            Group { children } => {
                let children: Vec<_> = children
                    .iter()
                    .flat_map(|child| self.build_rec(child, view, tr))
                    .collect();
                let bbox = children
                    .iter()
                    .map(|id| self.get(*id).bbox)
                    .fold(None, |acc, bbox| Some(bbox.union_opt(acc)))?;
                let item = PipelineItem::Group { children };
                Some(self.alloc(item, bbox, tr))
            }
            Clip {
                child,
                clip,
                units,
                fill_rule,
            } => {
                let clip_tr = match units {
                    Units::UserSpaceOnUse => tr,
                    Units::BoundingBox => {
                        let bbox = child.bbox(crate::Transform::identity())?;
                        tr * bbox.unit_transform()
                    }
                };
                let clip_bbox = view_apply(view, clip.bbox(clip_tr))?;
                let child_id = self.build_rec(child, Some(clip_bbox), tr)?;
                let bbox = clip_bbox.intersect(self.get(child_id).bbox)?;
                let item = PipelineItem::Clip {
                    child: child_id,
                    clip: clip.clone(),
                    clip_tr,
                    fill_rule: *fill_rule,
                };
                Some(self.alloc(item, bbox, tr))
            }
            Opacity { child, opacity } => {
                let child_id = self.build_rec(child, view, tr)?;
                let item = PipelineItem::Opacity {
                    child: child_id,
                    opacity: *opacity,
                };
                let bbox = self.get(child_id).bbox;
                Some(self.alloc(item, bbox, tr))
            }
            Transform {
                child,
                tr: child_tr,
            } => self.build_rec(child, view, tr * *child_tr),
        }
    }

    /// Allocate new pipeline node
    fn alloc(&mut self, item: PipelineItem, bbox: BBox, tr: Transform) -> PipelineNodeId {
        let id = PipelineNodeId(self.nodes.len());
        self.nodes.push(PipelineNode { item, bbox, tr });
        id
    }

    /// Get reference to the node by its id
    fn get(&self, id: PipelineNodeId) -> &PipelineNode {
        &self.nodes[id.0]
    }

    /// Id of the root node
    fn root(&self) -> Option<PipelineNodeId> {
        if self.nodes.is_empty() {
            None
        } else {
            Some(PipelineNodeId(self.nodes.len() - 1))
        }
    }

    /// Simple single pass rendering
    ///
    /// NOTE: `view` argument here is used to force layer to be of a specific size,
    ///       it is usually only used on the top level call.
    pub fn render(
        &self,
        rasterizer: &dyn Rasterizer,
        node_id: PipelineNodeId,
        view: Option<BBox>,
        bg: Option<LinColor>,
    ) -> Layer<LinColor> {
        // Force layer to be at least the size of the view
        let mut layer = Layer::new(view.unwrap_or_else(|| self.get(node_id).bbox), bg);
        self.render_rec(rasterizer, node_id, &mut layer);
        layer
    }

    fn render_rec(
        &self,
        rasterizer: &dyn Rasterizer,
        node_id: PipelineNodeId,
        layer: &mut Layer<LinColor>,
    ) {
        let node = self.get(node_id);

        use PipelineItem::*;
        match &node.item {
            Fill {
                path,
                paint,
                fill_rule,
            } => {
                // max bounds are not inclusive that is why we have `max + 1`
                let col_min = node.bbox.min().x().floor() as i32 - layer.x();
                let col_max = node.bbox.max().x().ceil() as i32 - layer.x() + 1;
                let row_min = node.bbox.min().y().floor() as i32 - layer.y();
                let row_max = node.bbox.max().y().ceil() as i32 - layer.y() + 1;
                // reduce the size of the image to only iterate over visible pixels
                let mut view = layer.view_mut(
                    row_min as usize,
                    row_max as usize,
                    col_min as usize,
                    col_max as usize,
                );
                // shift to account for view position
                let align = Transform::new_translate(
                    -node.bbox.min().x().floor(),
                    -node.bbox.min().y().floor(),
                );
                path.fill(rasterizer, align * node.tr, *fill_rule, paint, &mut view);
            }
            Group { children } => {
                for child in children {
                    self.render_rec(rasterizer, *child, layer);
                }
            }
            Opacity { child, opacity } => {
                let child_layer = self.render(rasterizer, *child, None, None);
                let opacity = *opacity as f32;
                layer.compose(&child_layer, |dst, src| {
                    let src = src * opacity;
                    dst.blend_over(&src)
                });
            }
            Clip {
                child,
                clip,
                clip_tr,
                fill_rule,
            } => {
                let mut mask = Layer::new(node.bbox, None);
                let align = Transform::new_translate(-mask.x() as Scalar, -mask.y() as Scalar);
                let mut child_layer = self.render(rasterizer, *child, None, None);
                clip.mask(rasterizer, align * *clip_tr, *fill_rule, mask.as_mut());
                // TOOD: do compose in a single pass?
                child_layer.compose(&mask, |dst, src| dst * (src as f32));
                layer.compose(&child_layer, |dst, src| dst.blend_over(&src));
            }
        }
    }
}

/// Image with top left corner at (x, y) coordinates
#[derive(Clone)]
pub struct Layer<C> {
    image: ImageOwned<C>,
    x: i32,
    y: i32,
}

impl<C> fmt::Debug for Layer<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Layer")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("w", &self.width())
            .field("h", &self.height())
            .finish()
    }
}

impl<C: Default + Copy> Layer<C> {
    /// New layer fully containing bounding box
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

    /// Create an empty layer
    pub fn empty() -> Self {
        Self {
            image: ImageOwned::empty(),
            x: 0,
            y: 0,
        }
    }

    /// X coordinate of the layer
    pub fn x(&self) -> i32 {
        self.x
    }

    /// Y coordinate of the layer
    pub fn y(&self) -> i32 {
        self.y
    }

    /// Translate layer by (dx, dy)
    pub fn translate(self, dx: i32, dy: i32) -> Self {
        Self {
            image: self.image,
            x: self.x + dx,
            y: self.y + dy,
        }
    }

    /// Compose other layer on top of this one taking into account offset
    pub fn compose<CF, CO>(&mut self, other: &Layer<CO>, compose: CF) -> &mut Self
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
        self
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

#[cfg(feature = "serde")]
mod serde_paint {
    use crate::{GradLinear, GradRadial, LinColor};

    use super::{Arc, Paint};
    use serde::{de, ser, Deserialize, Deserializer, Serialize, Serializer};
    use serde_json::Value;

    pub fn serialize<S>(paint: &Arc<dyn Paint>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        paint
            .to_json()
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<dyn Paint>, D::Error>
    where
        D: Deserializer<'de>,
    {
        match Value::deserialize(deserializer)? {
            Value::String(color) => {
                let color = color.parse::<LinColor>().map_err(de::Error::custom)?;
                Ok(Arc::new(color))
            }
            Value::Object(map) => {
                let paint_type = map
                    .get("type")
                    .ok_or_else(|| de::Error::missing_field("type"))?;
                match paint_type.as_str() {
                    Some(GradRadial::GRAD_TYPE) => {
                        let grad =
                            GradRadial::from_json(Value::Object(map)).map_err(de::Error::custom)?;
                        Ok(Arc::new(grad))
                    }
                    Some(GradLinear::GRAD_TYPE) => {
                        let grad =
                            GradLinear::from_json(Value::Object(map)).map_err(de::Error::custom)?;
                        Ok(Arc::new(grad))
                    }
                    _ => Err(de::Error::custom(format!(
                        "unknown paint type: {}",
                        paint_type
                    ))),
                }
            }
            value => Err(de::Error::custom(format!(
                "failed to parse paint: {}",
                value
            ))),
        }
    }
}

#[cfg(feature = "serde")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ActiveEdgeRasterizer;
    type Error = Box<dyn std::error::Error>;

    #[test]
    fn test_scene_view() -> Result<(), Error> {
        let rasterizer = ActiveEdgeRasterizer::default();

        let scene_str = r##"
        {
            "type": "transform",
            "tr": "translate(7, 7) rotate(45, 7, 7) scale(10)",
            "child": {
                "type": "fill",
                "paint": "#ff8040",
                "path": "M0,0 h1 v1 h-1 z"
            }
        }
        "##;
        let scene: Scene = serde_json::from_str(scene_str)?;

        // make sure that view is respected
        let img = scene.render(
            &rasterizer,
            Transform::identity(),
            Some(BBox::new((1.0, 2.0), (23.0, 23.0))),
            None,
        );
        assert_eq!(img.x(), 1);
        assert_eq!(img.y(), 2);
        assert_eq!(img.width(), 22);
        assert_eq!(img.height(), 21);

        // check image size with out view
        let img = scene.render(&rasterizer, Transform::identity(), None, None);
        assert_eq!(img.x(), 6);
        assert_eq!(img.y(), 4);
        assert_eq!(img.width(), 16);
        assert_eq!(img.height(), 15);

        Ok(())
    }
}
