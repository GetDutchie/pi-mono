import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

const repoRoot = process.cwd();

test("coding-agent keeps pi-server development-only like the upstream release", () => {
	const manifest = JSON.parse(readFileSync(join(repoRoot, "packages/coding-agent/package.json"), "utf8"));
	assert.equal(manifest.dependencies["@earendil-works/pi-server"], undefined);
	assert.equal(manifest.devDependencies["@earendil-works/pi-server"], "^0.99.2");
	assert.ok(manifest.files.includes("!dist/experimental"), "experimental server code is not in the published package");
});

test("publish cut treats pi-server, codemode, and mcp as unmodified upstream dependencies", () => {
	const scriptContent = readFileSync(join(repoRoot, "scripts/publish-dutchie-cut.mjs"), "utf8");
	for (const name of ["pi-server", "pi-codemode", "pi-mcp"]) {
		assert.ok(
			scriptContent.includes(`@earendil-works/${name}`) && scriptContent.includes(`packages/${name.replace("pi-", "")}`),
			`publish-dutchie-cut.mjs must validate ${name} against the upstream release`,
		);
	}
	assert.ok(
		!scriptContent.includes('upstreamName: "@earendil-works/pi-server"'),
		"pi-server must not be rescoped for a production coding-agent dependency",
	);
});

test("isolated cut validation checks production runtime dependencies, not development packages", () => {
	const scriptContent = readFileSync(join(repoRoot, "scripts/publish-dutchie-cut.mjs"), "utf8");
	assert.ok(scriptContent.includes("isolated-coding-agent-check"));
	assert.ok(scriptContent.includes('"@getdutchie/pi-ai", "@getdutchie/pi-agent-core", "@getdutchie/pi-tui"'));
	assert.ok(!scriptContent.includes('npm", ["ls", "@getdutchie/pi-server"'));
});
