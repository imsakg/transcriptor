<script lang="ts">
	import { IconEye, IconEyeOff } from '@tabler/icons-svelte';

	let hideShowKey = true;
</script>

<div class="container min-w-full h-full">
	<div class="grid grid-cols-5 p-4 gap-4">
		<div class="col-span-2 space-y-4">
			<div class="space-y-4 border border-1 p-4">
				<label class="label">
					<span>OpenAI Endpoint URL:</span>
					<div class="flex gap-2">
						<input
							class="input"
							title="Input (url)"
							type="url"
							placeholder="https://api.openai.com/v1/"
						/>
					</div>
				</label>

				<label class="label">
					<span>OpenAI API Key:</span>
					<div class="flex gap-2">
						<input
							class="input"
							title="Input (url)"
							type={hideShowKey ? 'password' : 'text'}
							placeholder="https://api.openai.com/v1/"
						/>
						<button
							type="button"
							class="btn-icon rounded-sm variant-filled"
							on:click={() => {
								hideShowKey = !hideShowKey;
							}}
							>{#if hideShowKey}
								<IconEye />
							{:else}
								<IconEyeOff />
							{/if}</button
						>
					</div>
				</label>

				<label class="label">
					<span>LLM Model Select:</span>
					<select class="select">
						<option value="1">ChatGPT 4 Turbo</option>
						<option value="2">ChatGPT 3.5</option>
					</select>
				</label>
			</div>

			<div class="space-y-4 border border-1 p-4">
				<label class="label">
					<span>Input Device:</span>
					<select class="select">
						<option value="1">Default Device</option>
					</select>
				</label>
				<label class="label">
					<span>Output Device:</span>
					<select class="select">
						<option value="1">Default Device</option>
					</select>
				</label>
			</div>
		</div>
		<div class="col-span-3 h-full">
			<div class="border border-1 p-4">
				<label class="label">
					<span>Instruction for LLM Model</span>
					<textarea class="textarea" rows="16" placeholder="Enter some long form content." />
				</label>
			</div>
		</div>
	</div>
</div>
