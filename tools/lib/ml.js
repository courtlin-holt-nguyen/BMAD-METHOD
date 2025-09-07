const fs = require('node:fs');
const path = require('node:path');
const { loadDataset, inferColumnTypes } = require('./data-loader');
const danfo = require('danfojs-node');
const { RandomForestClassifier, RandomForestRegression } = require('ml-random-forest');
const { DecisionTreeClassifier } = require('ml-cart');
const { StandardScaler } = require('ml-preprocess');
const { Matrix } = require('ml-matrix');
const ARIMA = require('arima');

function splitTrainTest(X, y, testSize = 0.2, stratify = false) {
  const n = X.length;
  const testN = Math.max(1, Math.min(Math.floor(n * testSize), Math.floor(n / 2)));
  const indices = Array.from({ length: n }, (_, i) => i);
  // simple shuffle
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const testIdx = new Set(indices.slice(0, testN));
  const X_train = [], X_test = [], y_train = [], y_test = [];
  for (let i = 0; i < n; i++) {
    if (testIdx.has(i)) {
      X_test.push(X[i]);
      y_test.push(y[i]);
    } else {
      X_train.push(X[i]);
      y_train.push(y[i]);
    }
  }
  return { X_train, X_test, y_train, y_test };
}

function toNumericMatrix(df, columns) {
  const data = df.loc({ columns }).values.map((row) => row.map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : Number.NaN)));
  return data;
}

function imputeNa(data, strategy = 'median') {
  const m = data.length;
  if (m === 0) return data;
  const n = data[0].length;
  const colStats = [];
  for (let j = 0; j < n; j++) {
    const colVals = [];
    for (let i = 0; i < m; i++) if (Number.isFinite(data[i][j])) colVals.push(data[i][j]);
    let fill = 0;
    if (colVals.length === 0) fill = 0;
    else if (strategy === 'mean') fill = colVals.reduce((a, b) => a + b, 0) / colVals.length;
    else {
      const s = [...colVals].sort((a, b) => a - b);
      fill = s[Math.floor(s.length / 2)];
    }
    colStats.push(fill);
  }
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (!Number.isFinite(data[i][j])) data[i][j] = colStats[j];
    }
  }
  return data;
}

function accuracyScore(yTrue, yPred) {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === yPred[i]) correct++;
  return correct / yTrue.length;
}

function r2Score(yTrue, yPred) {
  const mean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
  const ssTot = yTrue.reduce((acc, v) => acc + (v - mean) ** 2, 0);
  const ssRes = yTrue.reduce((acc, v, i) => acc + (v - yPred[i]) ** 2, 0);
  return 1 - ssRes / (ssTot || 1);
}

function rmse(yTrue, yPred) {
  const mse = yTrue.reduce((acc, v, i) => acc + (v - yPred[i]) ** 2, 0) / yTrue.length;
  return Math.sqrt(mse);
}

async function runRegression({ df, featureColumns, targetColumn, testSize }) {
  const Xraw = toNumericMatrix(df, featureColumns);
  const y = df[targetColumn].values.map((v) => (typeof v === 'number' ? v : Number(v)));
  const X = imputeNa(Xraw);

  const { X_train, X_test, y_train, y_test } = splitTrainTest(X, y, testSize);

  const rf = new RandomForestRegression({ nEstimators: 100, maxFeatures: Math.max(1, Math.floor(Math.sqrt(featureColumns.length))) });
  rf.train(X_train, y_train);
  const y_pred = rf.predict(X_test);
  const r2 = r2Score(y_test, y_pred);
  const error = rmse(y_test, y_pred);
  return { model: 'RandomForestRegression', metrics: { r2, rmse: error } };
}

async function runClassification({ df, featureColumns, targetColumn, testSize }) {
  const Xraw = toNumericMatrix(df, featureColumns);
  const y = df[targetColumn].values.map((v) => v);
  const X = imputeNa(Xraw);

  const { X_train, X_test, y_train, y_test } = splitTrainTest(X, y, testSize);

  const clf = new RandomForestClassifier({ nEstimators: 200, maxFeatures: Math.max(1, Math.floor(Math.sqrt(featureColumns.length))) });
  clf.train(X_train, y_train);
  const y_pred = clf.predict(X_test);
  const acc = accuracyScore(y_test, y_pred);
  return { model: 'RandomForestClassifier', metrics: { accuracy: acc } };
}

async function runForecast({ df, timeColumn, targetColumn, horizon }) {
  if (!df.columns.includes(timeColumn)) throw new Error(`Missing time column: ${timeColumn}`);
  const seriesDf = df.loc({ columns: [timeColumn, targetColumn] }).dropNa();
  const sorted = seriesDf.sortValues(timeColumn);
  const y = sorted[targetColumn].values.map((v) => (typeof v === 'number' ? v : Number(v)));
  const arima = new ARIMA({ p: 2, d: 1, q: 2, P: 1, D: 1, Q: 1, s: 12, method: 0, optimizer: 0 }).train(y);
  const [pred, errors] = arima.predict(horizon);
  return { model: 'ARIMA(2,1,2)(1,1,1)[12]', forecast: pred, stderr: errors };
}

async function runMl({ task, filePath, targetColumn, featureColumns, timeColumn, horizon = 12, testSize = 0.2, outputPath }) {
  const df = await loadDataset({ filePath });
  const inferred = inferColumnTypes(df);
  const allFeatures = df.columns.filter((c) => c !== targetColumn);
  const features = featureColumns && featureColumns.length ? featureColumns : allFeatures;

  const lines = [];
  lines.push(`# ML Workflow Report`);
  lines.push(`- **file**: ${path.basename(filePath)}`);
  lines.push(`- **task**: ${task}`);
  lines.push(`- **rows**: ${df.shape[0]}`);
  lines.push(`- **target**: ${targetColumn}`);
  lines.push(`- **features**: ${features.join(', ')}`);
  lines.push('');

  if (task === 'regression') {
    const { model, metrics } = await runRegression({ df, featureColumns: features, targetColumn, testSize });
    lines.push('## Results');
    lines.push(`- model: ${model}`);
    lines.push(`- r2: ${metrics.r2.toFixed(4)}`);
    lines.push(`- rmse: ${metrics.rmse.toFixed(4)}`);
  } else if (task === 'classification') {
    const { model, metrics } = await runClassification({ df, featureColumns: features, targetColumn, testSize });
    lines.push('## Results');
    lines.push(`- model: ${model}`);
    lines.push(`- accuracy: ${metrics.accuracy.toFixed(4)}`);
  } else if (task === 'forecast') {
    const result = await runForecast({ df, timeColumn, targetColumn, horizon });
    lines.push('## Results');
    lines.push(`- model: ${result.model}`);
    lines.push(`- horizon: ${horizon}`);
    lines.push(`- forecast: ${result.forecast.map((v) => Number(v).toFixed(4)).join(', ')}`);
  } else {
    throw new Error(`Unknown task: ${task}`);
  }

  const report = lines.join('\n');
  if (outputPath) fs.writeFileSync(outputPath, report, 'utf8');
  return report;
}

module.exports = { runMl };

