#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::style)]
#![allow(clippy::cast_precision_loss)] // u64 to f64
#![allow(clippy::cast_possible_truncation)] // u64 to usize
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::range_plus_one)]

use plotters::prelude::*;

pub struct Serie {
    pub label: String,
    pub values: Vec<(u64, f64)>,
}

pub fn draw(
    path: &str,
    caption: &str,
    series: &[Serie],
    resolution: (u32, u32),
    x_description: &str,
    y_description: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut min_x = u64::MAX;
    let mut max_x = 0;

    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for Serie { values, .. } in series {
        for &(x, y) in values {
            if x < min_x {
                min_x = x;
            }
            if max_x < x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if max_y < y {
                max_y = y;
            }
        }
    }

    let root = BitMapBackend::new(path, resolution).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 25).into_font())
        .margin(5)
        .x_label_area_size(50)
        .y_label_area_size(100)
        .build_cartesian_2d(
            (min_x - 1)..(max_x + 1),
            ((min_y / 2.)..(max_y * 2.)).log_scale(),
        )?;

    chart
        .configure_mesh()
        .x_desc(x_description)
        .y_desc(y_description)
        .y_label_formatter(&|y| format!("{y:+e}"))
        .draw()?;

    for (idx, Serie { label, values }) in series.iter().enumerate() {
        let color = Palette99::pick(idx).mix(0.9);

        chart
            .draw_series(LineSeries::new(values.clone(), color.filled()))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.filled()));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    Ok(())
}
