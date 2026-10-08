/**** ===========================================================================
 * PAPER 3 - SCRIPT 07a: IMERG half-hourly rainfall TRIGGER features (GEE)
 * ============================================================================
 *
 * WHY THIS EXISTS
 *   The current temporal features (x90 / seq) are MONTHLY MEANS of NDVI, NBR,
 *   BSI, SWIR and rainfall. A landslide is usually triggered by a single intense
 *   storm lasting hours. A monthly mean averages that storm away almost entirely:
 *   200 mm falling in 6 hours and 200 mm spread evenly over 30 days produce the
 *   SAME monthly value, but only the first causes failures. The ablations in this
 *   project showed the temporal branch contributes almost nothing once location
 *   leakage is removed (no_temporal 0.676 vs full 0.693) -- consistent with the
 *   temporal features carrying little trigger information. This script replaces
 *   monthly means with intensity/duration/antecedent statistics computed from
 *   HALF-HOURLY rainfall, which is the form the trigger actually takes.
 *
 * WHAT IT COMPUTES  (per point, per sample month -- see "WHY MONTH-ANCHORED")
 *   From IMERG half-hourly precipitationCal (mm/hr) over the sample month plus a
 *   45-day antecedent lead-in:
 *     max_30min        peak half-hourly rate in the month (mm/hr)
 *     max_1h/3h/6h/12h/24h   peak rolling-window accumulation in the month (mm)
 *     total_month      total rainfall in the month (mm)
 *     wet_hours        hours with rate > 0.5 mm/hr
 *     n_storms         count of distinct storm events (>= 4 consecutive wet
 *                      half-hours separated by >= 6 dry hours)
 *     max_storm_total  largest single-storm accumulation (mm)
 *     max_storm_dur_h  duration of that storm (hours)
 *     max_storm_int    mean intensity of that storm (mm/hr)
 *     ante_7d/15d/30d/45d  antecedent rainfall in the N days BEFORE the month
 *                      starts (mm) -- soil-wetness proxy; the 1D pore-pressure
 *                      diffusion approach of Paul et al. needs a window several
 *                      times the diffusive timescale, hence up to 45 days
 *     api_k0.9         antecedent precipitation index, daily decay k=0.9, at
 *                      month start
 *
 * WHY MONTH-ANCHORED, NOT EVENT-DAY-ANCHORED  (important -- read this)
 *   It is tempting to extract a 15-day pre-failure window around each landslide's
 *   exact date. That does NOT work for this dataset:
 *     - zhangjiajie_landslide_events.csv has exact dates (e.g. 2018-03-09), but
 *       only 49 of the source events behind the V2 samples are 'E' (event-dated).
 *     - 101 landslide sources are 'N' (NDVI-dated): month-level only.
 *     - ALL 968 negative samples are month-level only, by construction.
 *   Anchoring on exact dates would therefore produce features for ~a third of the
 *   positives and NONE of the negatives -- useless for classification, and worse,
 *   it would make positives and negatives structurally different (a positive would
 *   have a "real" window, a negative a made-up one), which is itself a leak: the
 *   model could learn "this row has event-style features => landslide".
 *   So every sample -- positive and negative alike -- gets the same treatment:
 *   statistics over its own (year, month) plus the 45 days before it. Identical
 *   feature definition for both classes, no structural giveaway.
 *
 * RESOLUTION CAVEAT (state this in the paper, do not hide it)
 *   IMERG is 0.1 degrees (~11 km). That is the SAME cell size as the `group`
 *   variable used for leakage control in this project. So all sample points inside
 *   one 0.1-deg cell in the same month receive IDENTICAL rainfall features. This
 *   is honest -- it is the true resolution of the data -- but it means these
 *   features cannot discriminate between nearby points within a cell, and they are
 *   constant within an independence cluster. The existing connected-component
 *   grouping already handles this correctly; no change to validation is needed.
 *   It also means: do NOT expect these features to separate two neighbouring
 *   slopes. Their value is separating WET-TRIGGER months from ordinary months at
 *   a given location, which is exactly what the season-matched V2 design isolates.
 *
 * INPUT
 *   An Earth Engine asset table of the points to extract, with columns:
 *     point (string/number id), lat, lon, year (number), month_num (number 1-12)
 *   Build it from samples_V2.npz with 07b_make_imerg_input.py (companion script),
 *   which writes imerg_input_V2.csv -- upload that CSV as a GEE table asset and
 *   put its asset id in POINTS_ASSET below.
 *
 * OUTPUT
 *   A CSV to Google Drive: imerg_features_V2.csv, one row per (point, year,
 *   month), joinable back onto the samples by (point, year, month).
 *
 * RUNTIME / SCALE NOTE
 *   Half-hourly IMERG is ~1,488 images per month. Extracting per (point, month)
 *   over ~1,400 unique point-months is a large but feasible GEE export; it is
 *   chunked by year below to avoid memory/time limits. If a chunk times out,
 *   lower CHUNK_MONTHS. Expect the export to take tens of minutes per year-chunk.
 * ========================================================================== */

// ---------------------------------------------------------------- CONFIG
var POINTS_ASSET = 'projects/zhangjiajie2/assets/imerg_input_V2';
var OUT_FOLDER   = 'GEE_exports';
var OUT_PREFIX   = 'imerg_features_V2';
var IMERG        = 'NASA/GPM_L3/IMERG_V07';   // half-hourly, 0.1 deg, 2000-present
var WET_MM_HR    = 0.5;    // rate above which a half-hour counts as "wet"
var STORM_MIN_HALFHOURS = 4;   // >= 2 h of wet half-hours to count as a storm
var STORM_GAP_HALFHOURS = 12;  // >= 6 h dry separates two storms
var ANTE_DAYS    = 45;
var YEARS        = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026];  // V2 samples span 2019-2026

var pts = ee.FeatureCollection(POINTS_ASSET);
print('Input point-months:', pts.size());
print('First row (check column names):', pts.first());

// ------------------------------------------------------- helper: month window
function monthRange(y, m) {
  var start = ee.Date.fromYMD(y, m, 1);
  return {start: start, end: start.advance(1, 'month')};
}

// ------------------------------------------------------- per point-month work
function featuresForPointMonth(feat) {
  var geom  = feat.geometry();
  var y     = ee.Number(feat.get('year'));
  var m     = ee.Number(feat.get('month_num'));
  var r     = monthRange(y, m);

  var month = ee.ImageCollection(IMERG)
      .filterDate(r.start, r.end)
      .filterBounds(geom)
      .select('precipitationCal');

  var ante = ee.ImageCollection(IMERG)
      .filterDate(r.start.advance(-ANTE_DAYS, 'day'), r.start)
      .filterBounds(geom)
      .select('precipitationCal');

  // -- point-level half-hourly series for the month (rate mm/hr) --
  var series = month.map(function (img) {
    var v = img.reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: geom,
      scale: 11132,          // IMERG native ~0.1 deg
      maxPixels: 1e9
    }).get('precipitationCal');
    return ee.Feature(null, {
      t: img.date().millis(),
      rate: ee.Algorithms.If(v, v, 0)
    });
  });
  var rates = ee.List(series.aggregate_array('rate'));     // mm/hr per 30-min slot
  var depths = rates.map(function (v) {                    // mm per 30-min slot
    return ee.Number(v).multiply(0.5);
  });

  var totalMonth = ee.Number(depths.reduce(ee.Reducer.sum()));
  var max30      = ee.Number(rates.reduce(ee.Reducer.max()));
  var wetSlots   = rates.map(function (v) {
    return ee.Number(ee.Number(v).gt(WET_MM_HR));
  });
  var wetHours   = ee.Number(wetSlots.reduce(ee.Reducer.sum())).multiply(0.5);

  // -- rolling accumulations, computed on the point series (cheap, list-based) --
  function rollMax(hours) {
    var w = ee.Number(hours).multiply(2).int();            // slots
    var n = depths.size();
    var starts = ee.List.sequence(0, n.subtract(w));
    var sums = starts.map(function (i) {
      i = ee.Number(i);
      return ee.Number(ee.List(depths.slice(i, i.add(w))).reduce(ee.Reducer.sum()));
    });
    return ee.Number(sums.reduce(ee.Reducer.max()));
  }

  // -- storm segmentation over the wet/dry run-length structure --
  // Build cumulative "storm id" by scanning: a dry gap >= STORM_GAP_HALFHOURS
  // closes a storm. Done with iterate to stay server-side.
  var stormStats = ee.Dictionary(ee.List(depths).iterate(function (d, acc) {
    acc = ee.Dictionary(acc);
    d = ee.Number(d);
    var isWet   = d.gt(ee.Number(WET_MM_HR).multiply(0.5));
    var curLen  = ee.Number(acc.get('curLen'));
    var curSum  = ee.Number(acc.get('curSum'));
    var dryRun  = ee.Number(acc.get('dryRun'));
    var best    = ee.Number(acc.get('bestSum'));
    var bestDur = ee.Number(acc.get('bestDur'));
    var nStorms = ee.Number(acc.get('nStorms'));

    // if wet: extend current storm, reset dry run
    var newLen  = ee.Algorithms.If(isWet, curLen.add(1), curLen);
    var newSum  = ee.Algorithms.If(isWet, curSum.add(d), curSum);
    var newDry  = ee.Algorithms.If(isWet, 0, dryRun.add(1));

    // if the dry run just reached the gap threshold, close the storm
    var closing = ee.Number(newDry).eq(STORM_GAP_HALFHOURS)
                    .and(ee.Number(newLen).gte(STORM_MIN_HALFHOURS));
    var closedBetter = closing.and(ee.Number(newSum).gt(best));
    return ee.Dictionary({
      curLen:  ee.Algorithms.If(ee.Number(newDry).eq(STORM_GAP_HALFHOURS), 0, newLen),
      curSum:  ee.Algorithms.If(ee.Number(newDry).eq(STORM_GAP_HALFHOURS), 0, newSum),
      dryRun:  newDry,
      bestSum: ee.Algorithms.If(closedBetter, newSum, best),
      bestDur: ee.Algorithms.If(closedBetter, ee.Number(newLen).multiply(0.5), bestDur),
      nStorms: ee.Algorithms.If(closing, nStorms.add(1), nStorms)
    });
  }, ee.Dictionary({curLen: 0, curSum: 0, dryRun: 0,
                    bestSum: 0, bestDur: 0, nStorms: 0})));

  // close a storm still open at month end
  var tailLen  = ee.Number(stormStats.get('curLen'));
  var tailSum  = ee.Number(stormStats.get('curSum'));
  var tailOk   = tailLen.gte(STORM_MIN_HALFHOURS);
  var bestSum  = ee.Number(ee.Algorithms.If(
                    tailOk.and(tailSum.gt(ee.Number(stormStats.get('bestSum')))),
                    tailSum, stormStats.get('bestSum')));
  var bestDur  = ee.Number(ee.Algorithms.If(
                    tailOk.and(tailSum.gt(ee.Number(stormStats.get('bestSum')))),
                    tailLen.multiply(0.5), stormStats.get('bestDur')));
  var nStorms  = ee.Number(stormStats.get('nStorms'))
                    .add(ee.Number(ee.Algorithms.If(tailOk, 1, 0)));

  // -- antecedent rainfall totals and API --
  function anteTotal(days) {
    var sub = ante.filterDate(r.start.advance(-days, 'day'), r.start);
    var v = sub.sum().multiply(0.5).reduceRegion({
      reducer: ee.Reducer.first(), geometry: geom, scale: 11132, maxPixels: 1e9
    }).get('precipitationCal');
    return ee.Algorithms.If(v, v, 0);
  }

  // API: sum over previous days of (daily mm * k^age). Daily aggregation first.
  var dayList = ee.List.sequence(1, ANTE_DAYS);
  var api = ee.Number(dayList.iterate(function (age, acc) {
    age = ee.Number(age);
    var dEnd   = r.start.advance(age.multiply(-1), 'day');
    var dStart = dEnd.advance(-1, 'day');
    var v = ee.ImageCollection(IMERG).filterDate(dStart, dEnd)
              .select('precipitationCal').sum().multiply(0.5)
              .reduceRegion({reducer: ee.Reducer.first(), geometry: geom,
                             scale: 11132, maxPixels: 1e9})
              .get('precipitationCal');
    var mm = ee.Number(ee.Algorithms.If(v, v, 0));
    return ee.Number(acc).add(mm.multiply(ee.Number(0.9).pow(age)));
  }, ee.Number(0)));

  return ee.Feature(null, {
    point:           feat.get('point'),
    lat:             feat.get('lat'),
    lon:             feat.get('lon'),
    year:            y,
    month_num:       m,
    total_month:     totalMonth,
    max_30min:       max30,
    max_1h:          rollMax(1),
    max_3h:          rollMax(3),
    max_6h:          rollMax(6),
    max_12h:         rollMax(12),
    max_24h:         rollMax(24),
    wet_hours:       wetHours,
    n_storms:        nStorms,
    max_storm_total: bestSum,
    max_storm_dur_h: bestDur,
    max_storm_int:   ee.Number(bestSum).divide(ee.Number(bestDur).max(0.5)),
    ante_7d:         anteTotal(7),
    ante_15d:        anteTotal(15),
    ante_30d:        anteTotal(30),
    ante_45d:        anteTotal(45),
    api_k09:         api
  });
}

// ------------------------------------------------------- chunked export by year
YEARS.forEach(function (yr) {
  var sub = pts.filter(ee.Filter.eq('year', yr));
  var out = sub.map(featuresForPointMonth);
  Export.table.toDrive({
    collection: out,
    description: OUT_PREFIX + '_' + yr,
    folder: OUT_FOLDER,
    fileNamePrefix: OUT_PREFIX + '_' + yr,
    fileFormat: 'CSV',
    selectors: ['point','lat','lon','year','month_num',
                'total_month','max_30min','max_1h','max_3h','max_6h','max_12h',
                'max_24h','wet_hours','n_storms','max_storm_total',
                'max_storm_dur_h','max_storm_int',
                'ante_7d','ante_15d','ante_30d','ante_45d','api_k09']
  });
});

print('Export tasks created -- open the Tasks tab and RUN each one.');
print('When all years finish, download the CSVs and run 07c_merge_imerg.py.');
