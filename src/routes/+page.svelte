<script lang="ts">
	import { IconX, IconEar } from '@tabler/icons-svelte';
	import { goto } from '$app/navigation';
	import { lastPage } from '$lib/Utils/stores';
</script>

<div class="container min-w-full h-full scroll-auto">
	<div class="grid grid-rows-6 p-4 gap-4 min-h-full">
		<div class="grid grid-cols-6 p-4 gap-4 min-w-full">
			<div
				class="cursor-pointer"
				on:click={() => {
					console.log('IconX clicked');
					$lastPage = '/';
					goto('/transcriptor');
				}}
			>
				<IconEar size="48px" />
			</div>
			<IconX />
			<IconX />
			<IconX />
			<IconX />
			<IconX />
		</div>
		<div class="grid grid-cols-6 p-4 gap-4 min-w-full">
			<IconX />
			<IconX />
			<IconX />
			<IconX />
			<IconX />
			<IconX />
		</div>
		<div class="grid grid-cols-6 p-4 gap-4 min-w-full">
			<IconX />
			<IconX />
			<IconX />
			<IconX />
			<IconX />
			<IconX />
		</div>
		<div class="grid grid-cols-6 p-4\ gap-4 min-w-full">
			<IconX />
			<IconX />
			<IconX />
			<IconX />
			<IconX />
			<IconX />
		</div>
		<div class="grid grid-cols-6 p-4 gap-4 min-w-full">
			<IconX />
			<IconX />
			<IconX />
			<IconX />
			<IconX />
			<IconX />
		</div>
		<div class="grid grid-cols-6 p-4 gap-4 min-w-full">
			<IconX />
			<IconX />
			<IconX />
			<IconX />
			<IconX />
			<IconX />
		</div>
	</div>
</div>
