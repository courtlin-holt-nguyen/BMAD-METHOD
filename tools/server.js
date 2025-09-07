const express = require('express');
const path = require('node:path');
const fs = require('node:fs');
const multer = require('multer');
const WebBuilder = require('./builders/web-builder');
const { runEda } = require('./lib/eda');
const { runMl } = require('./lib/ml');

const app = express();
app.use(express.json({ limit: '50mb' }));
const upload = multer({ dest: path.join(process.cwd(), 'uploads') });

// Health check
app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

// Build endpoints
app.post('/api/build', async (req, res) => {
  try {
    const { agentsOnly, teamsOnly, noExpansions, clean = true } = req.body || {};
    const builder = new WebBuilder({ rootDir: process.cwd() });
    if (clean) {
      await builder.cleanOutputDirs();
    }
    if (!teamsOnly) {
      await builder.buildAgents();
    }
    if (!agentsOnly) {
      await builder.buildTeams();
    }
    if (!noExpansions) {
      await builder.buildAllExpansionPacks({ clean: false });
    }
    res.json({ ok: true });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/api/build/expansions', async (req, res) => {
  try {
    const { name, clean = true } = req.body || {};
    const builder = new WebBuilder({ rootDir: process.cwd() });
    if (name) {
      await builder.buildExpansionPack(name, { clean });
    } else {
      await builder.buildAllExpansionPacks({ clean });
    }
    res.json({ ok: true });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

// List endpoints
app.get('/api/list/agents', async (_req, res) => {
  try {
    const builder = new WebBuilder({ rootDir: process.cwd() });
    const agents = await builder.resolver.listAgents();
    res.json({ ok: true, agents });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.get('/api/list/expansions', async (_req, res) => {
  try {
    const builder = new WebBuilder({ rootDir: process.cwd() });
    const expansions = await builder.listExpansionPacks();
    res.json({ ok: true, expansions });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

// Validate endpoint
app.post('/api/validate', async (_req, res) => {
  try {
    const builder = new WebBuilder({ rootDir: process.cwd() });
    const agents = await builder.resolver.listAgents();
    const teams = await builder.resolver.listTeams();
    const validated = { agents: [], teams: [] };
    for (const agent of agents) {
      await builder.resolver.resolveAgentDependencies(agent);
      validated.agents.push(agent);
    }
    for (const team of teams) {
      await builder.resolver.resolveTeamDependencies(team);
      validated.teams.push(team);
    }
    res.json({ ok: true, validated });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

// EDA endpoint (expects JSON body with filePath or raw data)
app.post('/api/eda', async (req, res) => {
  try {
    const { filePath, target, output, maxRows = 50_000, fileContent } = req.body || {};
    let absoluteFilePath = filePath;
    if (!absoluteFilePath && fileContent) {
      const tmpPath = path.join(process.cwd(), 'data', `eda-${Date.now()}.csv`);
      fs.mkdirSync(path.dirname(tmpPath), { recursive: true });
      fs.writeFileSync(tmpPath, fileContent, 'utf8');
      absoluteFilePath = tmpPath;
    }
    if (!absoluteFilePath || !fs.existsSync(absoluteFilePath)) {
      return res.status(400).json({ ok: false, error: 'Valid filePath or fileContent required' });
    }
    const report = await runEda({
      filePath: absoluteFilePath,
      targetColumn: target,
      outputPath: output,
      maxRows,
    });
    res.json({ ok: true, report });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

// ML endpoint (JSON body)
app.post('/api/ml', async (req, res) => {
  try {
    const {
      task,
      filePath,
      target,
      features,
      timeCol,
      horizon = 12,
      testSize = 0.2,
      output,
      fileContent,
    } = req.body || {};
    if (!task || !target) {
      return res.status(400).json({ ok: false, error: 'task and target are required' });
    }
    let absoluteFilePath = filePath;
    if (!absoluteFilePath && fileContent) {
      const tmpPath = path.join(process.cwd(), 'data', `ml-${Date.now()}.csv`);
      fs.mkdirSync(path.dirname(tmpPath), { recursive: true });
      fs.writeFileSync(tmpPath, fileContent, 'utf8');
      absoluteFilePath = tmpPath;
    }
    if (!absoluteFilePath || !fs.existsSync(absoluteFilePath)) {
      return res.status(400).json({ ok: false, error: 'Valid filePath or fileContent required' });
    }
    const featureColumns = Array.isArray(features)
      ? features
      : typeof features === 'string' && features
        ? features
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean)
        : undefined;
    const resultReport = await runMl({
      task,
      filePath: absoluteFilePath,
      targetColumn: target,
      featureColumns,
      timeColumn: timeCol,
      horizon,
      testSize,
      outputPath: output,
    });
    res.json({ ok: true, report: resultReport });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

// EDA file upload (multipart/form-data with field "file")
app.post('/api/eda/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ ok: false, error: 'file is required' });
    const { target, maxRows = 50_000 } = req.body || {};
    const tmpPath = req.file.path;
    const report = await runEda({ filePath: tmpPath, targetColumn: target, maxRows });
    res.json({ ok: true, report });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

// ML file upload (multipart/form-data with field "file")
app.post('/api/ml/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ ok: false, error: 'file is required' });
    const { task, target, features, timeCol, horizon = 12, testSize = 0.2 } = req.body || {};
    if (!task || !target)
      return res.status(400).json({ ok: false, error: 'task and target are required' });
    const featureColumns =
      typeof features === 'string' && features
        ? features
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean)
        : undefined;
    const tmpPath = req.file.path;
    const report = await runMl({
      task,
      filePath: tmpPath,
      targetColumn: target,
      featureColumns,
      timeColumn: timeCol,
      horizon: Number(horizon),
      testSize: Number(testSize),
    });
    res.json({ ok: true, report });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

// Serve static UI
const webRoot = path.join(process.cwd(), 'web');
app.use(express.static(webRoot));
app.get('*', (req, res) => {
  const indexPath = path.join(webRoot, 'index.html');
  if (fs.existsSync(indexPath)) {
    res.sendFile(indexPath);
  } else {
    res.status(404).send('UI not built');
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`BMAD web server listening on http://localhost:${port}`);
});
