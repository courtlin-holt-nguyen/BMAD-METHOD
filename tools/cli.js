const { Command } = require('commander');
const WebBuilder = require('./builders/web-builder');
const V3ToV4Upgrader = require('./upgraders/v3-to-v4-upgrader');
const IdeSetup = require('./installer/lib/ide-setup');
const path = require('node:path');
const pathFs = require('node:fs');
const { runEda } = require('./lib/eda');
const { runMl } = require('./lib/ml');

const program = new Command();

program
  .name('bmad-build')
  .description('BMAD-METHOD™ build tool for creating web bundles')
  .version('4.0.0');

program
  .command('build')
  .description('Build web bundles for agents and teams')
  .option('-a, --agents-only', 'Build only agent bundles')
  .option('-t, --teams-only', 'Build only team bundles')
  .option('-e, --expansions-only', 'Build only expansion pack bundles')
  .option('--no-expansions', 'Skip building expansion packs')
  .option('--no-clean', 'Skip cleaning output directories')
  .action(async (options) => {
    const builder = new WebBuilder({
      rootDir: process.cwd(),
    });

    try {
      if (options.clean) {
        console.log('Cleaning output directories...');
        await builder.cleanOutputDirs();
      }

      if (options.expansionsOnly) {
        console.log('Building expansion pack bundles...');
        await builder.buildAllExpansionPacks({ clean: false });
      } else {
        if (!options.teamsOnly) {
          console.log('Building agent bundles...');
          await builder.buildAgents();
        }

        if (!options.agentsOnly) {
          console.log('Building team bundles...');
          await builder.buildTeams();
        }

        if (!options.noExpansions) {
          console.log('Building expansion pack bundles...');
          await builder.buildAllExpansionPacks({ clean: false });
        }
      }

      console.log('Build completed successfully!');
    } catch (error) {
      console.error('Build failed:', error.message);
      process.exit(1);
    }
  });

program
  .command('build:expansions')
  .description('Build web bundles for all expansion packs')
  .option('--expansion <name>', 'Build specific expansion pack only')
  .option('--no-clean', 'Skip cleaning output directories')
  .action(async (options) => {
    const builder = new WebBuilder({
      rootDir: process.cwd(),
    });

    try {
      if (options.expansion) {
        console.log(`Building expansion pack: ${options.expansion}`);
        await builder.buildExpansionPack(options.expansion, { clean: options.clean });
      } else {
        console.log('Building all expansion packs...');
        await builder.buildAllExpansionPacks({ clean: options.clean });
      }

      console.log('Expansion pack build completed successfully!');
    } catch (error) {
      console.error('Expansion pack build failed:', error.message);
      process.exit(1);
    }
  });

program
  .command('list:agents')
  .description('List all available agents')
  .action(async () => {
    const builder = new WebBuilder({ rootDir: process.cwd() });
    const agents = await builder.resolver.listAgents();
    console.log('Available agents:');
    for (const agent of agents) console.log(`  - ${agent}`);
    process.exit(0);
  });

program
  .command('list:expansions')
  .description('List all available expansion packs')
  .action(async () => {
    const builder = new WebBuilder({ rootDir: process.cwd() });
    const expansions = await builder.listExpansionPacks();
    console.log('Available expansion packs:');
    for (const expansion of expansions) console.log(`  - ${expansion}`);
    process.exit(0);
  });

program
  .command('validate')
  .description('Validate agent and team configurations')
  .action(async () => {
    const builder = new WebBuilder({ rootDir: process.cwd() });
    try {
      // Validate by attempting to build all agents and teams
      const agents = await builder.resolver.listAgents();
      const teams = await builder.resolver.listTeams();

      console.log('Validating agents...');
      for (const agent of agents) {
        await builder.resolver.resolveAgentDependencies(agent);
        console.log(`  ✓ ${agent}`);
      }

      console.log('\nValidating teams...');
      for (const team of teams) {
        await builder.resolver.resolveTeamDependencies(team);
        console.log(`  ✓ ${team}`);
      }

      console.log('\nAll configurations are valid!');
    } catch (error) {
      console.error('Validation failed:', error.message);
      process.exit(1);
    }
  });

program
  .command('upgrade')
  .description('Upgrade a BMAD-METHOD™ V3 project to V4')
  .option('-p, --project <path>', 'Path to V3 project (defaults to current directory)')
  .option('--dry-run', 'Show what would be changed without making changes')
  .option('--no-backup', 'Skip creating backup (not recommended)')
  .action(async (options) => {
    const upgrader = new V3ToV4Upgrader();
    await upgrader.upgrade({
      projectPath: options.project,
      dryRun: options.dryRun,
      backup: options.backup,
    });
  });

// Exploratory Data Analysis command
program
  .command('eda')
  .description('Guide through exploratory data analysis of a provided dataset')
  .requiredOption('-f, --file <path>', 'Path to dataset file (CSV or JSON)')
  .option('-t, --target <column>', 'Optional target column for supervised problems')
  .option('-o, --output <path>', 'Optional path to write a Markdown EDA report')
  .option('--max-rows <n>', 'Limit rows to sample for speed (default 50000)', (v) => Number(v), 50000)
  .action(async (options) => {
    try {
      const absoluteFilePath = path.isAbsolute(options.file)
        ? options.file
        : path.join(process.cwd(), options.file);

      if (!pathFs.existsSync(absoluteFilePath)) {
        console.error(`Dataset not found: ${absoluteFilePath}`);
        process.exit(1);
      }

      const report = await runEda({
        filePath: absoluteFilePath,
        targetColumn: options.target,
        outputPath: options.output ? (path.isAbsolute(options.output) ? options.output : path.join(process.cwd(), options.output)) : undefined,
        maxRows: options.maxRows,
      });

      if (!options.output) {
        console.log('\n----- EDA SUMMARY (Markdown) -----\n');
        console.log(report);
      } else {
        console.log(`EDA report written to: ${options.output}`);
      }
    } catch (error) {
      console.error('EDA failed:', error.message);
      process.exit(1);
    }
  });

// Traditional ML workflows command
program
  .command('ml')
  .description('Run traditional ML workflows: regression, classification, or time-series forecasting')
  .requiredOption('-f, --file <path>', 'Path to dataset file (CSV or JSON)')
  .requiredOption('-k, --task <task>', 'Task: regression | classification | forecast')
  .requiredOption('-t, --target <column>', 'Target column name')
  .option('--features <cols>', 'Comma-separated feature column names (defaults to all except target)')
  .option('--time-col <column>', 'Time column for forecasting tasks')
  .option('--horizon <n>', 'Forecast horizon (default 12)', (v) => Number(v), 12)
  .option('--test-size <ratio>', 'Test set ratio between 0 and 0.5 (default 0.2)', (v) => Number(v), 0.2)
  .option('-o, --output <path>', 'Optional path to write a Markdown report of results')
  .action(async (options) => {
    try {
      const absoluteFilePath = path.isAbsolute(options.file)
        ? options.file
        : path.join(process.cwd(), options.file);

      if (!pathFs.existsSync(absoluteFilePath)) {
        console.error(`Dataset not found: ${absoluteFilePath}`);
        process.exit(1);
      }

      const features = options.features
        ? options.features.split(',').map((s) => s.trim()).filter(Boolean)
        : undefined;

      const resultReport = await runMl({
        task: options.task,
        filePath: absoluteFilePath,
        targetColumn: options.target,
        featureColumns: features,
        timeColumn: options.timeCol,
        horizon: options.horizon,
        testSize: options.testSize,
        outputPath: options.output ? (path.isAbsolute(options.output) ? options.output : path.join(process.cwd(), options.output)) : undefined,
      });

      if (!options.output) {
        console.log('\n----- ML RESULT SUMMARY (Markdown) -----\n');
        console.log(resultReport);
      } else {
        console.log(`ML results written to: ${options.output}`);
      }
    } catch (error) {
      console.error('ML workflow failed:', error.message);
      process.exit(1);
    }
  });

program.parse();
