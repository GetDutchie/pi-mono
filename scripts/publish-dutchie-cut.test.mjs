import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

const repoRoot = process.cwd();

test("packages/coding-agent declares @earendil-works/pi-server workspace dependency", () => {
	const manifest = JSON.parse(readFileSync(join(repoRoot, "packages/coding-agent/package.json"), "utf8"));
	assert.ok(manifest.dependencies["@earendil-works/pi-server"], "coding-agent must declare @earendil-works/pi-server dependency");
	assert.equal(manifest.dependencies["@earendil-works/pi-server"], "^0.85.0");
});

test("scripts/publish-dutchie-cut.mjs includes pi-server in rescoped packages", () => {
	const scriptContent = readFileSync(join(repoRoot, "scripts/publish-dutchie-cut.mjs"), "utf8");
	assert.ok(
		scriptContent.includes('upstreamName: "@earendil-works/pi-server", dutchieName: "@getdutchie/pi-server"'),
		"publish-dutchie-cut.mjs must include pi-server in packages",
	);
});

test("scripts/publish-dutchie-cut.mjs contains isolated consumer validation", () => {
	const scriptContent = readFileSync(join(repoRoot, "scripts/publish-dutchie-cut.mjs"), "utf8");
	assert.ok(
		scriptContent.includes("isolated-coding-agent-check"),
		"publish-dutchie-cut.mjs must test isolated direct coding-agent install",
	);
	assert.ok(
		scriptContent.includes('"@getdutchie/pi-server"') && scriptContent.includes('"ls"'),
		"publish-dutchie-cut.mjs must verify pi-server resolution via coding-agent",
	);
});
