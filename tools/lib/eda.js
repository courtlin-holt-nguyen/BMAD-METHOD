const fs = require('node:fs');
const path = require('node:path');
const danfo = require('danfojs-node');
const { loadDataset, inferColumnTypes } = require('./data-loader');

function describeNumeric(series) {
  const s = series.dropNa();
  const values = s.values.filter((v) => typeof v === 'number' && Number.isFinite(v));
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const q = (p) => {
    const idx = (sorted.length - 1) * p;
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    const w = idx - lo;
    return (1 - w) * sorted[lo] + w * sorted[hi];
  };
  const p25 = q(0.25);
  const p50 = q(0.5);
  const p75 = q(0.75);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (values.length - 1);
  const std = Math.sqrt(variance);
  return { count: values.length, mean, std, min, p25, p50, p75, max };
}

function valueCounts(series, maxValues = 20) {
  const s = series.dropNa();
  const map = new Map();
  for (const v of s.values) {
    map.set(v, (map.get(v) || 0) + 1);
  }
  const arr = [...map.entries()].sort((a, b) => b[1] - a[1]);
  return arr.slice(0, maxValues);
}

async function runEda({ filePath, targetColumn, outputPath, maxRows = 50000 }) {
  const df = await loadDataset({ filePath, maxRows });
  const columnToType = inferColumnTypes(df);

  const lines = [];
  lines.push(`# Exploratory Data Analysis Report`);
  lines.push(`- **file**: ${path.basename(filePath)}`);
  lines.push(`- **rows**: ${df.shape[0]}`);
  lines.push(`- **columns**: ${df.shape[1]}`);
  if (targetColumn) lines.push(`- **target**: ${targetColumn}`);
  lines.push('');

  lines.push('## Column Types');
  for (const [col, typ] of Object.entries(columnToType)) {
    lines.push(`- ${col}: ${typ}`);
  }
  lines.push('');

  lines.push('## Missing Values');
  for (const col of df.columns) {
    const missing = df[col].isNa().sum();
    const ratio = (missing / df.shape[0]) * 100;
    lines.push(`- ${col}: ${missing} (${ratio.toFixed(2)}%)`);
  }
  lines.push('');

  lines.push('## Numerical Summaries');
  for (const [col, typ] of Object.entries(columnToType)) {
    if (typ === 'number') {
      const stats = describeNumeric(df[col]);
      if (stats) {
        lines.push(`- ${col}: mean=${stats.mean.toFixed(4)}, std=${stats.std.toFixed(4)}, min=${stats.min}, p25=${stats.p25}, median=${stats.p50}, p75=${stats.p75}, max=${stats.max}`);
      }
    }
  }
  lines.push('');

  lines.push('## Categorical Value Counts (top 20)');
  for (const [col, typ] of Object.entries(columnToType)) {
    if (typ === 'string' || typ === 'boolean') {
      const counts = valueCounts(df[col]);
      const total = df.shape[0];
      const items = counts.map(([value, count]) => `${value} (${((count / total) * 100).toFixed(2)}%)`).join(', ');
      lines.push(`- ${col}: ${items}`);
    }
  }
  lines.push('');

  if (targetColumn && df.columns.includes(targetColumn)) {
    lines.push('## Target Relationships (quick signals)');
    for (const [col, typ] of Object.entries(columnToType)) {
      if (col === targetColumn) continue;
      if (typ === 'number' && columnToType[targetColumn] === 'number') {
        const xSeries = df[col];
        const ySeries = df[targetColumn];
        const xs = [];
        const ys = [];
        for (let i = 0; i < df.shape[0]; i++) {
          const xv = xSeries.values[i];
          const yv = ySeries.values[i];
          if (
            typeof xv === 'number' && Number.isFinite(xv) &&
            typeof yv === 'number' && Number.isFinite(yv)
          ) {
            xs.push(xv);
            ys.push(yv);
          }
        }
        const n = xs.length;
        if (n >= 2) {
          const meanX = xs.reduce((a, b) => a + b, 0) / n;
          const meanY = ys.reduce((a, b) => a + b, 0) / n;
          let num = 0;
          let denX = 0;
          let denY = 0;
          for (let i = 0; i < n; i++) {
            const dx = xs[i] - meanX;
            const dy = ys[i] - meanY;
            num += dx * dy;
            denX += dx * dx;
            denY += dy * dy;
          }
          const denom = Math.sqrt(denX * denY);
          if (denom > 0) {
            const corr = num / denom;
            if (Number.isFinite(corr)) {
              lines.push(`- corr(${col}, ${targetColumn}) = ${corr.toFixed(4)}`);
            }
          }
        }
      }
    }
    lines.push('');
  }

  lines.push('## Next-step Guidance');
  lines.push('- Consider removing high-missing columns (>40%) or imputing missing values.');
  lines.push('- Scale numerical features if using distance-based models; encode categoricals.');
  lines.push('- For forecasting, ensure `time` column is parsed as date and sorted.');
  lines.push('- For classification, check class imbalance and consider stratified splits.');

  const report = lines.join('\n');
  if (outputPath) {
    fs.writeFileSync(outputPath, report, 'utf8');
  }
  return report;
}

module.exports = { runEda };

