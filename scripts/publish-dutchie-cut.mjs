#!/usr/bin/env node

import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const packages = [
	{ directory: "packages/ai", upstreamName: "@earendil-works/pi-ai", dutchieName: "@getdutchie/pi-ai" },
	{ directory: "packages/tui", upstreamName: "@earendil-works/pi-tui", dutchieName: "@getdutchie/pi-tui" },
	{ directory: "packages/agent", upstreamName: "@earendil-works/pi-agent-core", dutchieName: "@getdutchie/pi-agent-core" },
	{
		directory: "packages/session-backends/sqlite-node",
		upstreamName: "@earendil-works/pi-session-backend-sqlite-node",
		dutchieName: "@getdutchie/pi-session-backend-sqlite-node",
	},
	{ directory: "packages/server", upstreamName: "@earendil-works/pi-server", dutchieName: "@getdutchie/pi-server" },
	{ directory: "packages/coding-agent", upstreamName: "@earendil-works/pi-coding-agent", dutchieName: "@getdutchie/pi-coding-agent" },
];

/**
 * Workspace packages that are deliberately NOT rescoped: the fork does not
 * modify them, so a rescoped package may depend on the public
 * `@earendil-works/*` release at the same version. Each entry maps the package
 * name to its directory so the staging step can PROVE it is still identical to
 * upstream before trusting public resolution.
 *
 * None of these may depend on pi-ai, or a consumer would end up with the fork's
 * @getdutchie/pi-ai and a second upstream copy in the same tree.
 */
const externalUpstreamPackages = new Map([
	["@earendil-works/pi-telemetry", "packages/telemetry"],
	["@earendil-works/pi-client", "packages/client"],
	["@earendil-works/pi-protocol", "packages/protocol"],
	["@earendil-works/chord", "packages/chord"],
]);

const rewriteExtensions = new Set([
	".cjs",
	".cts",
	".d.cts",
	".d.mts",
	".d.ts",
	".js",
	".json",
	".md",
	".mjs",
	".mts",
	".ts",
	".tsx",
]);
const ignoredDirectories = new Set([".git", "node_modules"]);
const packageNameMap = new Map(packages.map((pkg) => [pkg.upstreamName, pkg.dutchieName]));
const upstreamInternalNames = packages.map((pkg) => pkg.upstreamName);
const UPSTREAM_SPECIFIER_PATTERN = /["'](@earendil-works\/[^"'/]+)(?:\/[^"']*)?["']/g;

/**
 * Package names imported by a staged file. Only quoted specifiers count: the
 * @earendil-works scope also shows up in prose and in globs like
 * `@earendil-works/pi-*` inside comments that survive into dist.
 */
function importedUpstreamPackages(content) {
	const names = new Set();
	for (const match of content.matchAll(UPSTREAM_SPECIFIER_PATTERN)) names.add(match[1]);
	return names;
}

function log(message = "") {
	process.stdout.write(`${message}\n`);
}

function printUsage() {
	log(`Usage: node scripts/publish-dutchie-cut.mjs --version <x.y.z-dutchie.n> [options]

Builds private @getdutchie/* package tarballs from the upstream-named monorepo
without mutating the checkout. The staging copy rewrites package names,
internal dependency declarations, and built import specifiers from
@earendil-works/* to @getdutchie/*, then validates an isolated install.

Options:
  --version <version>      Required Dutchie prerelease version, e.g. 0.82.1-dutchie.2
  --out <dir>              Output directory. Defaults to a new temp directory
  --force                  Remove --out first if it already exists
  --skip-build             Do not run npm run build:offline before staging
  --skip-install-check     Do not validate the tarballs in an isolated install
  --publish                Publish staged tarballs to GitHub Packages after validation
  --registry <url>         Registry for publish. Defaults to https://npm.pkg.github.com
  --tag <tag>              Publish tag. Defaults to latest
  --help                   Show this help
`);
}

function parseArgs() {
	const options = {
		force: false,
		installCheck: true,
		outDir: undefined,
		publish: false,
		registry: "https://npm.pkg.github.com",
		skipBuild: false,
		tag: "latest",
		version: undefined,
	};
	const args = process.argv.slice(2);

	for (let i = 0; i < args.length; i++) {
		const arg = args[i];
		if (arg === "--help") {
			printUsage();
			process.exit(0);
		}
		if (arg === "--force") {
			options.force = true;
			continue;
		}
		if (arg === "--skip-build") {
			options.skipBuild = true;
			continue;
		}
		if (arg === "--skip-install-check") {
			options.installCheck = false;
			continue;
		}
		if (arg === "--publish") {
			options.publish = true;
			continue;
		}
		if (["--out", "--registry", "--tag", "--version"].includes(arg)) {
			const value = args[++i];
			if (!value) throw new Error(`${arg} requires a value`);
			if (arg === "--out") options.outDir = value;
			if (arg === "--registry") options.registry = value;
			if (arg === "--tag") options.tag = value;
			if (arg === "--version") options.version = value;
			continue;
		}
		throw new Error(`Unknown option: ${arg}`);
	}

	if (!options.version) throw new Error("--version is required");
	if (!/^\d+\.\d+\.\d+-dutchie\.\d+$/.test(options.version)) {
		throw new Error(`--version must look like x.y.z-dutchie.n, got ${options.version}`);
	}

	return options;
}

function commandForPlatform(command) {
	return process.platform === "win32" ? `${command}.cmd` : command;
}

function run(command, args, options = {}) {
	const displayedArgs = options.displayArgs ?? args;
	log(`$ ${[command, ...displayedArgs].join(" ")}`);
	const result = spawnSync(commandForPlatform(command), args, {
		cwd: options.cwd,
		encoding: "utf8",
		stdio: options.capture ? ["inherit", "pipe", "pipe"] : "inherit",
	});
	if (result.status !== 0) {
		const output = [result.stdout, result.stderr].filter(Boolean).join("\n");
		throw new Error(output ? `Command failed: ${command} ${args.join(" ")}\n${output}` : `Command failed: ${command} ${args.join(" ")}`);
	}
	return result;
}

function readJson(path) {
	try {
		return JSON.parse(readFileSync(path, "utf8"));
	} catch (error) {
		throw new Error(`Failed to parse ${path}: ${error instanceof Error ? error.message : String(error)}`);
	}
}

function writeJson(path, value) {
	writeFileSync(path, `${JSON.stringify(value, null, "\t")}\n`);
}

function isInsidePath(child, parent) {
	const relativePath = relative(parent, child);
	return relativePath === "" || (!relativePath.startsWith("..") && !relativePath.startsWith("/"));
}

function prepareOutputDirectory(options, repoRoot) {
	const outDir = options.outDir ? resolve(options.outDir) : mkdtempSync(join(tmpdir(), "pi-dutchie-cut-"));
	if (isInsidePath(outDir, repoRoot)) {
		throw new Error(`Output directory must be outside the repository: ${outDir}`);
	}
	if (existsSync(outDir)) {
		if (!options.force && options.outDir) throw new Error(`Output directory already exists. Use --force to replace it: ${outDir}`);
		rmSync(outDir, { force: true, recursive: true });
	}
	mkdirSync(outDir, { recursive: true });
	return outDir;
}

function rewritePackageJson(packageJson, pkg, version) {
	if (packageJson.name !== pkg.upstreamName) {
		throw new Error(`${pkg.directory}/package.json has name ${packageJson.name}, expected ${pkg.upstreamName}`);
	}
	packageJson.name = pkg.dutchieName;
	packageJson.version = version;

	for (const section of ["dependencies", "devDependencies", "optionalDependencies", "peerDependencies"]) {
		const dependencies = packageJson[section];
		if (!dependencies) continue;
		const rewritten = {};
		for (const [name, specifier] of Object.entries(dependencies)) {
			const dutchieName = packageNameMap.get(name);
			if (dutchieName) {
				rewritten[dutchieName] = version;
			} else {
				rewritten[name] = specifier;
			}
		}
		packageJson[section] = rewritten;
	}
}

function shouldRewriteFile(path) {
	for (const extension of rewriteExtensions) {
		if (path.endsWith(extension)) return true;
	}
	return false;
}

function rewriteTextFile(path) {
	let content = readFileSync(path, "utf8");
	let rewritten = content;
	for (const [upstreamName, dutchieName] of packageNameMap) {
		rewritten = rewritten.split(upstreamName).join(dutchieName);
	}
	if (rewritten !== content) writeFileSync(path, rewritten);
}

function walkFiles(directory, callback) {
	for (const entry of readdirSyncSafe(directory)) {
		const path = join(directory, entry.name);
		if (entry.isDirectory()) {
			if (!ignoredDirectories.has(entry.name)) walkFiles(path, callback);
			continue;
		}
		if (entry.isFile()) callback(path);
	}
}

function readdirSyncSafe(directory) {
	return existsSync(directory) ? readdirSync(directory, { withFileTypes: true }) : [];
}

function removeShrinkwrap(stageDirectory) {
	// The upstream coding-agent shrinkwrap pins @earendil-works tarballs from the
	// public npm registry. A private rescoped cut must not ship that lockfile: it
	// would override the rewritten @getdutchie dependencies and recreate the bug.
	rmSync(join(stageDirectory, "npm-shrinkwrap.json"), { force: true });
}

function validateNoUpstreamInternalSpecifiers(stageDirectory, pkg) {
	const scannedFiles = [];
	for (const relativePath of ["package.json", "npm-shrinkwrap.json", "dist"]) {
		const absolutePath = join(stageDirectory, relativePath);
		if (!existsSync(absolutePath)) continue;
		const stats = statSync(absolutePath);
		if (stats.isDirectory()) {
			walkFiles(absolutePath, (file) => {
				if (shouldRewriteFile(file)) scannedFiles.push(file);
			});
		} else {
			scannedFiles.push(absolutePath);
		}
	}

	const failures = [];
	for (const file of scannedFiles) {
		const content = readFileSync(file, "utf8");
		for (const upstreamName of upstreamInternalNames) {
			if (content.includes(upstreamName)) failures.push(`${relative(stageDirectory, file)} still contains ${upstreamName}`);
		}
		// Anything else in the @earendil-works scope must be a package this script
		// KNOWS is unmodified by the fork. Without this, an upstream merge that
		// introduces a new workspace dependency ships silently: the rescoped
		// package would resolve it from public npm, and if the fork ever patches
		// that package the cut would quietly run upstream's code instead.
		for (const specifier of importedUpstreamPackages(content)) {
			if (packageNameMap.has(specifier) || externalUpstreamPackages.has(specifier)) continue;
			failures.push(
				`${relative(stageDirectory, file)} imports unknown workspace package ${specifier}. ` +
					"Add it to `packages` to rescope it, or to `externalUpstreamPackages` if the fork does not modify it.",
			);
		}
	}
	if (failures.length > 0) {
		const unique = [...new Set(failures)];
		throw new Error(`${pkg.dutchieName} staging validation failed:\n${unique.map((failure) => `  - ${failure}`).join("\n")}`);
	}
}

/**
 * Prove that every package we let resolve from public npm really is identical to
 * upstream at this commit. If the fork starts patching one of them, the cut must
 * rescope it instead of silently shipping upstream's code.
 */
function verifyExternalUpstreamPackagesAreUnmodified(repoRoot) {
	const revParse = spawnSync(commandForPlatform("git"), ["rev-parse", "--verify", "--quiet", "upstream/main"], {
		cwd: repoRoot,
		encoding: "utf8",
	});
	if (revParse.status !== 0) {
		log("WARNING: no upstream/main ref, cannot verify that unrescoped workspace packages match upstream.");
		log("         Run `git fetch upstream` to enable this check.");
		return;
	}

	const modified = [];
	for (const [name, directory] of externalUpstreamPackages) {
		const diff = spawnSync(commandForPlatform("git"), ["diff", "--quiet", "upstream/main", "--", directory], {
			cwd: repoRoot,
			encoding: "utf8",
		});
		if (diff.status !== 0) modified.push(`${name} (${directory})`);
	}
	if (modified.length > 0) {
		throw new Error(
			`These packages are treated as unmodified upstream dependencies but differ from upstream/main:\n${modified
				.map((entry) => `  - ${entry}`)
				.join("\n")}\nRescope them in \`packages\` instead of resolving them from public npm.`,
		);
	}
	log(`Verified ${externalUpstreamPackages.size} unrescoped workspace packages match upstream/main.`);
}

function stagePackage(pkg, paths, version) {
	const sourceDirectory = join(paths.repoRoot, pkg.directory);
	const stageDirectory = join(paths.stageRoot, pkg.dutchieName.replace("@getdutchie/", ""));
	if (!existsSync(join(sourceDirectory, "package.json"))) {
		throw new Error(
			`${pkg.directory}/package.json does not exist. Upstream may have moved or renamed ${pkg.upstreamName}; update the \`packages\` list.`,
		);
	}
	if (!existsSync(join(sourceDirectory, "dist"))) {
		throw new Error(`${pkg.directory}/dist does not exist. Run npm run build:offline first, or omit --skip-build.`);
	}

	cpSync(sourceDirectory, stageDirectory, {
		recursive: true,
		filter: (source) => !source.split(/[\\/]/).some((part) => ignoredDirectories.has(part)),
	});

	const packageJsonPath = join(stageDirectory, "package.json");
	const packageJson = readJson(packageJsonPath);
	rewritePackageJson(packageJson, pkg, version);
	writeJson(packageJsonPath, packageJson);

	if (pkg.upstreamName === "@earendil-works/pi-coding-agent") removeShrinkwrap(stageDirectory);

	walkFiles(stageDirectory, (file) => {
		if (shouldRewriteFile(file)) rewriteTextFile(file);
	});
	validateNoUpstreamInternalSpecifiers(stageDirectory, pkg);
	return stageDirectory;
}

function packPackage(stageDirectory, tarballDirectory) {
	const output = run("npm", ["pack", "--json", "--ignore-scripts", "--pack-destination", tarballDirectory], {
		capture: true,
		cwd: stageDirectory,
	}).stdout;
	let packed;
	try {
		packed = JSON.parse(output)[0];
	} catch (error) {
		throw new Error(`Failed to parse npm pack JSON for ${stageDirectory}: ${error instanceof Error ? error.message : String(error)}`);
	}
	const tarball = join(tarballDirectory, packed.filename);
	log(`  ${packed.filename}: ${packed.files.length} files, ${packed.size} bytes packed`);
	return tarball;
}

function fileSpecifier(fromDirectory, file) {
	const relativePath = relative(fromDirectory, file).replaceAll("\\", "/");
	return `file:${relativePath.startsWith(".") ? relativePath : `./${relativePath}`}`;
}

function validateInstall(paths, tarballsByName) {
	const installDirectory = join(paths.outDir, "install-check");
	mkdirSync(installDirectory, { recursive: true });
	const dependencies = Object.fromEntries(
		packages.map((pkg) => [pkg.dutchieName, fileSpecifier(installDirectory, tarballsByName.get(pkg.dutchieName))]),
	);
	writeJson(join(installDirectory, "package.json"), { private: true, type: "module", dependencies });
	writeFileSync(join(installDirectory, ".npmrc"), "@getdutchie:registry=https://npm.pkg.github.com\n");
	run("npm", ["install", "--omit=dev", "--ignore-scripts"], { cwd: installDirectory });
	const ls = run("npm", ["ls", "@getdutchie/pi-ai", "@getdutchie/pi-agent-core", "@getdutchie/pi-tui", "@getdutchie/pi-coding-agent", "--all"], {
		capture: true,
		cwd: installDirectory,
	}).stdout;
	log(ls.trim());
	run(process.execPath, [
		"--input-type=module",
		"-e",
		'import("@getdutchie/pi-coding-agent").then(() => console.log("@getdutchie/pi-coding-agent import ok"))',
	], { cwd: installDirectory });
}

function publishTarball(tarball, options) {
	const args = ["publish", tarball, "--registry", options.registry, "--tag", options.tag, "--ignore-scripts"];
	if (options.registry === "https://npm.pkg.github.com") {
		const token = run("gh", ["auth", "token"], { capture: true }).stdout.trim();
		args.push(`--//npm.pkg.github.com/:_authToken=${token}`);
	}
	run("npm", args, {
		displayArgs: args.map((arg) => (arg.includes("_authToken=") ? "--//npm.pkg.github.com/:_authToken=<redacted>" : arg)),
	});
}

function main() {
	const options = parseArgs();
	const repoRoot = process.cwd();
	const rootPackageJson = readJson(join(repoRoot, "package.json"));
	if (rootPackageJson.name !== "pi-monorepo") {
		throw new Error("Run this script from the repository root");
	}

	const outDir = prepareOutputDirectory(options, repoRoot);
	const paths = {
		outDir,
		repoRoot,
		stageRoot: join(outDir, "stage"),
		tarballDirectory: join(outDir, "tarballs"),
	};
	mkdirSync(paths.stageRoot, { recursive: true });
	mkdirSync(paths.tarballDirectory, { recursive: true });

	verifyExternalUpstreamPackagesAreUnmodified(repoRoot);

	if (!options.skipBuild) {
		run("npm", ["run", "build:offline"], { cwd: repoRoot });
	}

	const tarballsByName = new Map();
	for (const pkg of packages) {
		log(`\nStaging ${pkg.upstreamName} -> ${pkg.dutchieName}@${options.version}`);
		const stageDirectory = stagePackage(pkg, paths, options.version);
		const tarball = packPackage(stageDirectory, paths.tarballDirectory);
		tarballsByName.set(pkg.dutchieName, tarball);
	}

	if (options.installCheck) {
		log("\nValidating isolated install from staged tarballs...");
		validateInstall(paths, tarballsByName);
	}

	if (options.publish) {
		log(`\nPublishing to ${options.registry} with tag ${options.tag}...`);
		for (const pkg of packages) publishTarball(tarballsByName.get(pkg.dutchieName), options);
	} else {
		log("\nDry run complete. Tarballs are ready but were not published.");
	}

	log(`\nDutchie cut artifacts: ${outDir}`);
}

try {
	main();
} catch (error) {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
}
