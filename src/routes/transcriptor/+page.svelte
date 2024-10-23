<!-- YOU CAN DELETE EVERYTHING IN THIS PAGE -->
<script lang="ts">
	import { lastPage } from '$lib/Utils/stores';
	import { goto } from '$app/navigation';
	import { listen } from '@tauri-apps/api/event';
	import { invoke } from '@tauri-apps/api/core';

	let output = '';
	let outputs = [];
	let input = '';
	let inputs = [];

	function sendOutput() {
		console.log('js: js2rs: ' + output);
		outputs.push({ timestamp: Date.now(), message: output });
		invoke('js2rs', { message: output });
	}

	// listen('rs2js', (event) => {
	// 	console.log('js: rs2js: ' + event);
	// 	let input = event.payload;
	// 	inputs.push({ timestamp: Date.now(), message: input });
	// });

	listen('whisper_in', (event) => {
		input = event.payload + '\n' + input;
		inputs.push({ timestamp: Date.now(), message: event.payload });
	});

	listen('whisper_out', (event) => {
		output = event.payload + '\n' + output;
		outputs.push({ timestamp: Date.now(), message: event.payload });
	});
</script>

<div class="container min-w-full h-full overflow-hidden pb-24">
	<div class="pl-2 pt-1 cursor-pointer">
		<a
			on:click={() => {
				goto($lastPage);
			}}
			><u> Previous Page </u>
		</a>
	</div>
	<div class="grid grid-rows-2 p-4 gap-2 h-full space-y-6">
		<div>
			<div class="grid grid-cols-2 gap-4 h-full">
				<div class="space-y-4">
					<label class="label h-full">
						<span>Input Source</span>
						<textarea
							bind:value={input}
							class="textarea h-full"
							placeholder="Enter some long form content."
						/>
					</label>
				</div>
				<div class="space-y-4 h-full">
					<label class="label h-full">
						<span>Output Source</span>
						<textarea
							bind:value={output}
							class="textarea h-full"
							placeholder="Enter some long form content."
						/>
					</label>
				</div>
			</div>
		</div>

		<div class="h-full">
			<label class="label h-full">
				<span>LLM's Response</span>
				<textarea class="textarea box-border h-full" placeholder="Enter some long form content." />
			</label>
		</div>
	</div>
</div>
