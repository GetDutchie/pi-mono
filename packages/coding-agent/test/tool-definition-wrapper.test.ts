/**
 * Tests for tool-definition-wrapper: constrainedSampling propagation between
 * ToolDefinition and AgentTool in both directions.
 */

import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { ConstrainedSamplingConfig } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import type { ToolDefinition } from "../src/core/extensions/types.ts";
import { createToolDefinitionFromAgentTool, wrapToolDefinition } from "../src/core/tools/tool-definition-wrapper.ts";

const requireStrict: ConstrainedSamplingConfig = { type: "json_schema", strict: "require" };

function makeDefinition(constrainedSampling?: false | ConstrainedSamplingConfig): ToolDefinition<any, unknown> {
	return {
		name: "example",
		label: "Example",
		description: "An example tool",
		parameters: Type.Object({ value: Type.String() }),
		...(constrainedSampling !== undefined ? { constrainedSampling } : {}),
		execute: async () => ({ content: [], isError: false, details: undefined }),
	};
}

function makeAgentTool(constrainedSampling?: false | ConstrainedSamplingConfig): AgentTool<any> {
	return {
		name: "example",
		label: "Example",
		description: "An example tool",
		parameters: Type.Object({ value: Type.String() }),
		...(constrainedSampling !== undefined ? { constrainedSampling } : {}),
		execute: async () => ({ content: [], isError: false, details: undefined }),
	};
}

describe("tool-definition-wrapper constrainedSampling propagation", () => {
	it("wrapToolDefinition omits constrainedSampling for an ordinary ToolDefinition", () => {
		const agentTool = wrapToolDefinition(makeDefinition());
		expect(agentTool.constrainedSampling).toBeUndefined();
	});

	it("wrapToolDefinition carries a config through to AgentTool", () => {
		const agentTool = wrapToolDefinition(makeDefinition(requireStrict));
		expect(agentTool.constrainedSampling).toEqual(requireStrict);
	});

	it("wrapToolDefinition carries an explicit false through to AgentTool", () => {
		const agentTool = wrapToolDefinition(makeDefinition(false));
		expect(agentTool.constrainedSampling).toBe(false);
	});

	it("createToolDefinitionFromAgentTool omits constrainedSampling for an ordinary AgentTool", () => {
		const definition = createToolDefinitionFromAgentTool(makeAgentTool());
		expect(definition.constrainedSampling).toBeUndefined();
	});

	it("createToolDefinitionFromAgentTool carries a config back to ToolDefinition", () => {
		const definition = createToolDefinitionFromAgentTool(makeAgentTool(requireStrict));
		expect(definition.constrainedSampling).toEqual(requireStrict);
	});
});
