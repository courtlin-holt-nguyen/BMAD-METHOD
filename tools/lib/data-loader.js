const fs = require('node:fs');
const path = require('node:path');
const danfo = require('danfojs-node');

async function loadDataset({ filePath, maxRows }) {
  const ext = path.extname(filePath).toLowerCase();
  let df;
  if (ext === '.csv') {
    df = await danfo.readCSV(filePath, { lowMemoryMode: true });
  } else if (ext === '.json') {
    const raw = fs.readFileSync(filePath, 'utf8');
    const json = JSON.parse(raw);
    df = new danfo.DataFrame(json);
  } else {
    throw new Error(`Unsupported file type: ${ext}`);
  }

  if (typeof maxRows === 'number' && Number.isFinite(maxRows) && df.shape[0] > maxRows) {
    df = df.sample(maxRows, { seed: 1337 });
  }
  return df;
}

function inferColumnTypes(df) {
  const columnToType = {};
  for (const col of df.columns) {
    const series = df[col];
    const nonNull = series.dropNa();
    let inferred = 'unknown';
    if (nonNull.values.length === 0) {
      inferred = 'empty';
    } else {
      const sample = nonNull.values.slice(0, 50);
      const isNumber = sample.every((v) => typeof v === 'number' && Number.isFinite(v));
      const isBoolean = sample.every((v) => typeof v === 'boolean');
      const isDate = sample.every((v) => v instanceof Date || (typeof v === 'string' && !Number.isNaN(Date.parse(v))));
      if (isNumber) inferred = 'number';
      else if (isBoolean) inferred = 'boolean';
      else if (isDate) inferred = 'date';
      else inferred = 'string';
    }
    columnToType[col] = inferred;
  }
  return columnToType;
}

module.exports = {
  loadDataset,
  inferColumnTypes,
};

