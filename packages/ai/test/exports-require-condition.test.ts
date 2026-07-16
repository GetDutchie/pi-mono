import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// Regression guard for a recurring dutchie-fork bug: the package.json `exports`
// map keeps losing its CJS `require` condition on subpaths across releases
// (dutchie.9 had it, dutchie.10 dropped it again). Any subpath that declares
// an `import` condition MUST also declare a `require` condition, otherwise
// CJS consumers doing `require("@earendil-works/pi-ai/<subpath>")` throw
// ERR_PACKAGE_PATH_NOT_EXPORTED.
const packageJsonPath = fileURLToPath(new URL("../package.json", import.meta.url));
const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8")) as {
	exports?: Record<string, unknown>;
};

describe("package.json exports", () => {
	it("declares a require condition for every subpath that declares an import condition", () => {
		const exportsMap = packageJson.exports;
		expect(exportsMap).toBeDefined();
		if (!exportsMap) return;

		const subpaths = Object.keys(exportsMap);
		expect(subpaths.length).toBeGreaterThan(0);

		for (const subpath of subpaths) {
			const entry = exportsMap[subpath];
			if (entry === null || typeof entry !== "object" || Array.isArray(entry)) {
				continue;
			}

			const conditions = entry as Record<string, unknown>;
			if (!("import" in conditions)) {
				continue;
			}

			expect(
				"require" in conditions,
				`exports['${subpath}'] has 'import' but is missing 'require' — CJS consumers (require()) will throw ERR_PACKAGE_PATH_NOT_EXPORTED. Every subpath with import MUST also have require (dutchie fork regression 2026-07-15).`,
			).toBe(true);
		}
	});
});
