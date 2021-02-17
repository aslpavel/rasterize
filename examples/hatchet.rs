#![deny(warnings)]

use rasterize::*;
use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashMap},
    hash::{Hash, Hasher},
    io::Write,
};

const EPSILON: Scalar = 1e-6;

// intersection between a dividing line and a segment
#[derive(Debug, Clone, Copy)]
struct Intersection {
    // segment under intersection
    segment: Segment,
    // segment id (subpath_index, segment_index)
    segment_id: (usize, usize),
    // parameter `t` of the intersection
    segment_t: Scalar,
    // whether to follow segminet or line
    segment_follow: bool,
    // winding delta introduced by intersection
    winding_delta: i32,
    // winding number following intersection
    winding: i32,
    // x coordinate of the intersection
    x_coord: Scalar,
    // intersection with low_y or high_y line
    y_low: bool,
    // index starting from intersections with lower x coordinate
    index: usize,
}

impl Ord for Intersection {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.index, self.y_low).cmp(&(other.index, other.y_low))
    }
}

impl PartialOrd for Intersection {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for Intersection {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.y_low.hash(state);
    }
}

impl PartialEq for Intersection {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.y_low == other.y_low
    }
}

impl Eq for Intersection {}

// Intersect path specified by segments with `y = 0` when `tr` transformation is applied.
fn intersect(segments: &Vec<Vec<Segment>>, tr: Transform, y_low: bool) -> Vec<Intersection> {
    let mut ints = Vec::new();
    for (subpath_index, subpath) in segments.iter().enumerate() {
        for (segment_index, segment) in subpath.iter().enumerate() {
            let segment_id = (subpath_index, segment_index);
            let segment_tr = segment.transform(tr);
            for root in segment_tr.roots() {
                let deriv_y = segment_tr.deriv().at(root).y();
                if deriv_y.abs() < EPSILON {
                    // throw away tangent intersections
                    continue;
                }
                ints.push(Intersection {
                    segment: *segment,
                    segment_id,
                    segment_t: root,
                    segment_follow: if y_low { deriv_y > 0.0 } else { deriv_y < 0.0 },
                    winding_delta: if deriv_y < 0.0 { -1 } else { 1 },
                    winding: 0,
                    x_coord: segment_tr.at(root).x(),
                    y_low,
                    index: 0,
                });
            }
        }
    }

    // order intersections by `x` coordinate
    ints.sort_by(|a, b| {
        a.x_coord
            .partial_cmp(&b.x_coord)
            .expect("invalid coordinate value")
    });

    let mut winding = 0;
    let mut output: Vec<Intersection> = Vec::with_capacity(ints.len());
    for mut int in ints {
        // remove duplicate intersections
        if output.len() > 0 {
            let int_prev = output[output.len() - 1];
            if (int.x_coord - int_prev.x_coord).abs() < EPSILON {
                if int.winding_delta + int_prev.winding_delta == 0 {
                    // this might happend when line touches the corner of the path
                    winding -= int_prev.winding_delta;
                    output.pop();
                }
                continue;
            }
        }
        // update winding and index
        winding += int.winding_delta;
        int.winding = winding;
        int.index = output.len();
        output.push(int);
    }
    output
}

/// Convert path to hatched version of the path
///
/// All hatches are perpendicular to `normal` with perioud of `normal.lenght()`,
/// with covered width equal to `period * ratio`.
fn hatch(path: &Path, normal: Line, ratio: Scalar) -> Path {
    if ratio <= 0.0 || ratio >= 1.0 {
        return path.clone();
    }
    let period = normal.length();

    // find transformation which makes hatch lines horizontal
    let dir = normal.direction();
    let tangent = Line::new(
        normal.start(),
        normal.start() + Point::new(-dir.y(), dir.x()),
    );
    let tr = Transform::make_horizontal(tangent);

    // find transformation so origin would be the first line effecting the path
    let bbox_tr = match path.bbox(tr) {
        None => return path.clone(),
        Some(bbox) => bbox,
    };
    let offset_y = -(bbox_tr.y() / period).floor() * period;
    let tr = Transform::default().translate(-bbox_tr.x(), offset_y) * tr;

    // all segmentes grouped by subpaths with included closing lines
    let segments: Vec<_> = path
        .subpaths()
        .iter()
        .map(|subpath| {
            let last = subpath
                .end()
                .is_close_to(subpath.start())
                .then(|| Line::new(subpath.end(), subpath.start()).into());
            subpath
                .segments()
                .iter()
                .copied()
                .chain(last)
                .collect::<Vec<_>>()
        })
        .collect();

    let mut subpaths_out = Vec::new();
    let mut offset_y = 0.0;
    while offset_y < bbox_tr.height() + period {
        let ints_low = intersect(
            &segments,
            Transform::default().translate(0.0, -offset_y) * tr,
            true,
        );
        let ints_high = intersect(
            &segments,
            Transform::default().translate(0.0, -offset_y - period * ratio) * tr,
            false,
        );

        // mapping from segment_id to a list of intersections ordered by curves `t` parameter
        let mut id_to_ints = HashMap::new();
        for int in ints_low.iter().copied().chain(ints_high.iter().copied()) {
            id_to_ints
                .entry(int.segment_id)
                .or_insert_with(Vec::new)
                .push(int);
        }
        for (_, ints) in id_to_ints.iter_mut() {
            // order intersections by `t`
            ints.sort_by(|a, b| {
                a.segment_t
                    .partial_cmp(&b.segment_t)
                    .expect("invalid curve parameter value")
            });
        }

        let mut unvisited: BTreeSet<_> = ints_low
            .iter()
            .copied()
            .chain(ints_high.iter().copied())
            .collect();
        while !unvisited.is_empty() {
            let int_start = match unvisited.range(..).take(1).next() {
                Some(int_start) => *int_start,
                None => break,
            };
            let mut subpath_out = Vec::new();
            let mut int = int_start;
            while unvisited.contains(&int) {
                unvisited.remove(&int);

                if int.segment_follow {
                    // check if current segment have more intersections with higher `t`
                    let mut int_next = None;
                    for int_other in id_to_ints[&int.segment_id].iter() {
                        if int_other.segment_t > int.segment_t {
                            let seg = int.segment.cut(int.segment_t, int_other.segment_t);
                            subpath_out.push(seg);
                            int_next = Some(*int_other);
                            break;
                        }
                    }

                    match int_next {
                        Some(int_next) => {
                            int = int_next;
                        }
                        None => {
                            let (_, seg) = int.segment.split_at(int.segment_t);
                            subpath_out.push(seg);

                            // find and add all not intersecting segments
                            let mut segment_id = int.segment_id;
                            let start_id = segment_id;
                            loop {
                                let (subpath_index, segment_index) = segment_id;
                                let subpath = &segments[subpath_index];
                                let segment_index = (segment_index + 1) % subpath.len();
                                segment_id = (subpath_index, segment_index);
                                if segment_id == start_id {
                                    break;
                                }
                                if let Some(ints) = id_to_ints.get(&segment_id) {
                                    int = *ints
                                        .first()
                                        .expect("id to intersections map contains empty list");
                                    let (seg, _) = int.segment.split_at(int.segment_t);
                                    subpath_out.push(seg);
                                    break;
                                }
                                subpath_out.push(subpath[segment_index]);
                            }
                        }
                    }
                } else {
                    let index = if int.winding == 0 {
                        // going in the direction of lower x
                        int.index - 1
                    } else if int.winding == int.winding_delta {
                        // going in the direction of higher x
                        int.index + 1
                    } else {
                        panic!(
                            "path with winding not in [-1..1] are not supported: {:?}",
                            int
                        );
                    };
                    let int_next = if int.y_low {
                        ints_low[index]
                    } else {
                        ints_high[index]
                    };
                    let line = Line::new(
                        int.segment.at(int.segment_t),
                        int_next.segment.at(int_next.segment_t),
                    );
                    subpath_out.push(line.into());
                    int = int_next;
                }
                if int == int_start {
                    break;
                }
            }
            if let Some(subpath) = SubPath::new(subpath_out, true) {
                subpaths_out.push(subpath);
            }
        }

        // TODO:
        //   - include subpaths, when bounding box fits between low and high lines
        offset_y += period;
    }

    Path::new(subpaths_out)
}

fn generate_bar(
    frac: Scalar,
    bbox: BBox,
    border: Scalar,
    radii: Point,
    hatch_normal: Line,
    hatch_ratio: Scalar,
) -> Path {
    let frac = frac.clamp(0.0, 1.0);
    let mut path = Path::builder()
        .move_to((bbox.x(), bbox.y()))
        .rbox((bbox.width(), bbox.height()), radii)
        .build();
    if frac < 1.0 {
        let bx = bbox.x() + border;
        let by = bbox.y() + border;
        let bh = (bbox.height() - 2.0 * border) * (1.0 - frac);
        let bw = bbox.width() - 2.0 * border;
        let border = Path::builder()
            .move_to((bx, by))
            .rbox((bw, bh), radii)
            .build()
            .reverse();
        path.extend(border);
        let ib = hatch_normal.length() * hatch_ratio;
        let hatch_path = Path::builder()
            .move_to((bx + ib, by + ib))
            .rbox((bw - 2.0 * ib, bh - 2.0 * ib), radii)
            .build();
        path.extend(hatch(&hatch_path, hatch_normal, hatch_ratio));
    }
    path
}

struct Glyph {
    path: Path,
    name: String,
    unicode: char,
}

fn generate_font(
    out: &mut dyn Write,
    family: &str,
    glyphs: impl IntoIterator<Item = Glyph>,
) -> Result<(), std::io::Error> {
    writeln!(out, "<?xml version=\"1.0\" standalone=\"no\"?>")?;
    writeln!(out, "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\" >")?;
    writeln!(out, "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" version=\"1.1\"> ")?;
    writeln!(out, "<defs>")?;
    writeln!(out, "  <font horiz-adv-x=\"500\">")?;
    writeln!(out, "  <font-face")?;
    writeln!(out, "    font-family=\"{}\"", family)?;
    writeln!(out, "    units-per-em=\"1000\"")?;
    writeln!(out, "    ascent=\"800\"")?;
    writeln!(out, "    descent=\"-200\"")?;
    writeln!(out, "  />")?;
    for glyph in glyphs {
        writeln!(
            out,
            "<glyph glyph-name=\"{}\" unicode=\"{}\"  d=\"{}\" />",
            glyph.name,
            glyph.unicode,
            glyph.path.to_svg_path(),
        )?;
    }
    writeln!(out, "  </font>")?;
    writeln!(out, "</defs>")?;
    writeln!(out, "</svg>")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut glyphs = Vec::new();
    let tr = Transform::default().translate(0.0, 1000.0).scale(1.0, -1.0);
    let offset = '0' as u32;
    for index in 0..11 {
        let mut path = generate_bar(
            index as f64 / 10.0,
            BBox::new((0.0, 0.0), (460.0, 1200.0)),
            70.0,
            Point::new(50.0, 50.0),
            Line::new((0.0, 0.0), (70.0, 70.0)),
            0.3,
        );
        path.transform(tr);
        glyphs.push(Glyph {
            path,
            name: format!("bar-{}", index),
            unicode: std::char::from_u32(offset + index).unwrap(),
        });
    }

    let mut out = Vec::new();
    generate_font(&mut out, "Bars", glyphs)?;
    println!("{}", String::from_utf8_lossy(&out));

    Ok(())
}
