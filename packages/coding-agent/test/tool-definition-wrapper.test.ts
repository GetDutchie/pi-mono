/**
 * Tests for tool-definition-wrapper: strict flag propagation between
 * ToolDefinition and AgentTool in both directions.
 */

import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import type { ToolDefinition } from "../src/core/extensions/types.ts";
import { createToolDefinitionFromAgentTool, wrapToolDefinition } from "../src/core/tools/tool-definition-wrapper.ts";

function makeDefinition(strict?: boolean): ToolDefinition<any, unknown> {
	return {
		name: "example",
		label: "Example",
		description: "An example tool",
		parameters: Type.Object({ value: Type.String() }),
		...(strict !== undefined ? { strict } : {}),
		execute: async () => ({ content: [], isError: false, details: undefined }),
	};
}

function makeAgentTool(strict?: boolean): AgentTool<any> {
	return {
		name: "example",
		label: "Example",
		description: "An example tool",
		parameters: Type.Object({ value: Type.String() }),
		...(strict !== undefined ? { strict } : {}),
		execute: async () => ({ content: [], isError: false, details: undefined }),
	};
}

describe("tool-definition-wrapper strict propagation", () => {
	it("wrapToolDefinition omits strict for an ordinary ToolDefinition", () => {
		const agentTool = wrapToolDefinition(makeDefinition());
		expect(agentTool.strict).toBeUndefined();
	});

	it("wrapToolDefinition carries strict:true through to AgentTool", () => {
		const agentTool = wrapToolDefinition(makeDefinition(true));
		expect(agentTool.strict).toBe(true);
	});

	it("wrapToolDefinition carries explicit strict:false through to AgentTool", () => {
		const agentTool = wrapToolDefinition(makeDefinition(false));
		expect(agentTool.strict).toBe(false);
	});

	it("createToolDefinitionFromAgentTool omits strict for an ordinary AgentTool", () => {
		const definition = createToolDefinitionFromAgentTool(makeAgentTool());
		expect(definition.strict).toBeUndefined();
	});

	it("createToolDefinitionFromAgentTool carries strict:true back to ToolDefinition", () => {
		const definition = createToolDefinitionFromAgentTool(makeAgentTool(true));
		expect(definition.strict).toBe(true);
	});
});
