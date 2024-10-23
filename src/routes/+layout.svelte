<script lang="ts">
	import '../app.pcss';
	//Importing Skeleton's components
	import {
		AppShell,
		AppBar,
		AppRail,
		AppRailAnchor,
		LightSwitch,
		initializeStores,
		FileButton,
		Modal,
		getModalStore,
		popup,
		storePopup
	} from '@skeletonlabs/skeleton';

	// Highlight JS
	import hljs from 'highlight.js/lib/core';
	import 'highlight.js/styles/github-dark.css';
	import { storeHighlightJs } from '@skeletonlabs/skeleton';
	import xml from 'highlight.js/lib/languages/xml'; // for HTML
	import css from 'highlight.js/lib/languages/css';
	import javascript from 'highlight.js/lib/languages/javascript';
	import typescript from 'highlight.js/lib/languages/typescript';

	hljs.registerLanguage('xml', xml); // for HTML
	hljs.registerLanguage('css', css);
	hljs.registerLanguage('javascript', javascript);
	hljs.registerLanguage('typescript', typescript);
	storeHighlightJs.set(hljs);

	// Floating UI for Popups
	import { computePosition, autoUpdate, flip, shift, offset, arrow } from '@floating-ui/dom';
	storePopup.set({ computePosition, autoUpdate, flip, shift, offset, arrow });

	import { invoke } from '@tauri-apps/api/core';
	import { getCurrent } from '@tauri-apps/api/window';

	import { page } from '$app/stores';

	//Importing Tabler's icon pack
	import {
		IconLayout,
		IconHome,
		IconSettings,
		IconMenu2,
		IconTerminal2,
		IconChecklist,
		IconInfoOctagon,
		IconAlignBoxRightMiddle,
		IconBattery4,
		IconBattery3,
		IconBattery2,
		IconBattery1,
		IconBatteryOff,
		IconSatellite,
		IconAntenna,
		IconAntennaOff,
		IconCircleCheck,
		IconAlertHexagon,
		IconX,
		IconRotate2,
		IconRotateClockwise2
	} from '@tabler/icons-svelte';

	// Necessary importations for navbar's transition
	import { slide } from 'svelte/transition';
	import { quadInOut, quintIn, quintInOut, quintOut } from 'svelte/easing';

	// Importing variables from stores.js
	import {
		leftNavActive,
		rightBarActive,
		heartbeat,
		last_heartbeat,
		uav_batteryVoltage,
		uav_gpsStatus,
		uav_networkStatus,
		pingStatus,
		raspberryBoot,
		pixhawkBoot,
		GPSTest,
		lidarTest,
		pitotTest,
		cameraTest,
		IMUTest,
		motorTest,
		notepadText
	} from '$lib/Utils/stores';

	import { onMount } from 'svelte';
	import { autoModeWatcher } from '@skeletonlabs/skeleton';

	// Initializing the stores
	initializeStores();

	//Assigning getModalStore function to modalStore variable
	const modalStore = getModalStore();

	//Configurations for Exit Modal
	import type { ModalSettings } from '@skeletonlabs/skeleton';
	const exitModal: ModalSettings = {
		type: 'confirm',
		// Data
		title: 'Confirm Exit',
		body: 'Are you sure you want to exit?',
		// TRUE if confirm pressed, FALSE if cancel pressed
		/**
		 * @param {boolean} r
		 */
		response: (r: any) => {
			console.log('Response: ', r);
			if (r) invoke('exit_app');
		}
	};

	onMount(() => {
		autoModeWatcher();
	});
</script>

<Modal />
<!-- App Shell -->
<AppShell scrollbarGutter="stable" regionPage="">
	<!-- Header -->
	<svelte:fragment slot="header">
		<AppBar
			gridColumns="grid-cols-3"
			slotDefault="place-self-center"
			padding="p-0"
			spacing="space-y-0"
			slotTrail="place-content-end"
			shadow="shadow-2xl drop-shadow-2xl"
		>
			<!-- Left Section of Header -->
			<svelte:fragment slot="lead">
				<!-- Left Section's Outer Container -->
				<div class="flex flex-row justify-between w-full place-content-center place-items-center">
					<!-- (1) Container for Navbar Button -->
					<div
						class="flex flex-row cursor-pointer w-14 h-14 hover:bg-primary-hover-token place-content-center place-items-center"
						on:click={() => {
							$leftNavActive = !$leftNavActive;
						}}
					>
						<IconMenu2 class="w-8 h-8" />
					</div>

					<!-- Don't show the Status Section at Home or About Pages -->
					{#if $page.url.pathname !== '/' && $page.url.pathname !== '/about'}
						<!-- (2) Status Section's Container -->
						<div
							class="flex flex-row space-x-6 w-14 h-14 lg:mr-20 md:m-0 sm:m-0 place-content-end place-items-center"
						></div>
					{/if}
				</div>
			</svelte:fragment>

			<!-- Middle Section of Header -->
			<!-- Middle Section's Outer Container -->
			<div class="flex flex-row items-center p-0 m-0 place-content-center checked:">
				<!-- async drag start -->
				<p
					on:mousedown={async () => {
						// while mouse down

						console.log('dragging');
						// if mouse up
						// import resizeDirection
						await getCurrent().startDragging();
						console.log('dragging end');
					}}
					class="pt-1 pb-0 pl-0 pr-0 m-0 text-2xl text-center border-8 border-none cursor-default text-slate-700 dark:text-slate-300 active:cursor-grab hover:cursor-grab focus"
				>
					transcriptor
				</p>
			</div>

			<!-- Right Section of Header -->
			<svelte:fragment slot="trail">
				<!-- Right Section's Outer Container (Don't show the Status Section and Databar Button, show only Exit Button if you're in Home or About Page) -->
				<div
					class={`flex flex-row place-content-center place-items-center ${
						$page.url.pathname !== '/' && $page.url.pathname !== '/about'
							? 'justify-between w-full h-full'
							: 'cursor-pointer w-14 h-14 hover:bg-primary-token'
					}`}
				>
					{#if $page.url.pathname !== '/' && $page.url.pathname !== '/about'}
						<!-- (1) Status Section -->
						<div
							class="flex flex-row w-full h-full space-x-6 lg:ml-20 md:m-0 sm:m-0 place-content-start place-items-center"
						>
							<!-- Latency Status -->
							<div class="flex flex-row">
								<!-- Adjusting the Latency view depending on the latency value with conditional rendering and styling -->
							</div>

							<!-- Heartbeat Status -->
							<div class="flex flex-row">
								<!-- Adjusting the Hearbeat view depending on the Heartbeat's boolean value -->
							</div>
						</div>

						<!-- (2) Exit and Databar Buttons -->
						<div class="flex flex-row">
							<!-- Databar Button -->
							<div
								class="flex flex-row cursor-pointer w-14 h-14 hover:bg-primary-hover-token place-content-center place-items-center"
								on:click={() => {
									$rightBarActive = !$rightBarActive;
								}}
							>
								<IconAlignBoxRightMiddle class="w-8 h-8" />
							</div>

							<!-- Exit Button -->
							<div
								class="flex flex-row cursor-pointer w-14 h-14 place-content-center place-items-center"
							>
								<button
									on:click={() => {
										console.log('Exit Button Pressed');
										modalStore.trigger(exitModal);
									}}
									class="flex w-full h-full cursor-pointer hover:bg-red-700 dark:hover:bg-red-900 place-content-center place-items-center"
								>
									<IconX class="w-8 h-8" />
								</button>
							</div>
						</div>
					{:else}
						<!-- Exit Button -->
						<button
							class="flex w-full h-full cursor-pointer hover:bg-red-800 place-content-center place-items-center"
							on:click={() => {
								console.log('Exit Button Pressed');

								modalStore.trigger(exitModal);
							}}
						>
							<IconX class="w-8 h-8" />
						</button>
					{/if}
				</div>
			</svelte:fragment>
		</AppBar>
	</svelte:fragment>

	<!-- App Rail -->
	<!-- Navbar -->
	<svelte:fragment slot="sidebarLeft">
		<!-- Activate the Navbar if Navbar Button pressed -->
		{#if $leftNavActive}
			<!-- Navbar's Outer Container -->
			<div
				transition:slide={{ delay: 20, duration: 300, easing: quintOut, axis: 'x' }}
				class="h-full"
			>
				<!-- AppRail Component that let us make a Navbar -->
				<AppRail width="w-14">
					<!-- Top Section of Navbar -->
					<svelte:fragment slot="lead">
						<!-- Home Page Button -->
						<div use:popup={{ event: 'hover', target: 'homePageTooltip', placement: 'right' }}>
							<AppRailAnchor
								href="/"
								selected={$page.url.pathname === '/'}
								active="bg-[#e0e8f6] dark:bg-[#22324a] "
							>
								<svelte:fragment slot="lead">
									<IconHome />
								</svelte:fragment>
							</AppRailAnchor>
						</div>

						<!-- Dashboard Page Button -->
						<div use:popup={{ event: 'hover', target: 'dashboardPageTooltip', placement: 'right' }}>
							<AppRailAnchor
								href="/dashboard"
								active="bg-[#e0e8f6] dark:bg-[#22324a]"
								selected={$page.url.pathname === '/dashboard'}
							>
								<svelte:fragment slot="lead">
									<IconLayout />
								</svelte:fragment>
							</AppRailAnchor>
						</div>

						<!-- Telemetry Page Button -->
						<div use:popup={{ event: 'hover', target: 'telemetryPageTooltip', placement: 'right' }}>
							<AppRailAnchor
								href="/telemetry"
								active="bg-[#e0e8f6] dark:bg-[#22324a]"
								selected={$page.url.pathname === '/telemetry'}
							>
								<svelte:fragment slot="lead">
									<IconTerminal2 />
								</svelte:fragment>
							</AppRailAnchor>
						</div>

						<!-- Test Page Button -->
						<div use:popup={{ event: 'hover', target: 'testPageTooltip', placement: 'right' }}>
							<AppRailAnchor
								href="/test"
								active="bg-[#e0e8f6] dark:bg-[#22324a]"
								selected={$page.url.pathname === '/test'}
							>
								<svelte:fragment slot="lead">
									<IconChecklist />
								</svelte:fragment>
							</AppRailAnchor>
						</div>
					</svelte:fragment>

					<!-- Bottom Section of Navbar -->
					<svelte:fragment slot="trail">
						<!-- About Page Button -->
						<div use:popup={{ event: 'hover', target: 'aboutPageTooltip', placement: 'right' }}>
							<AppRailAnchor
								href="/about"
								active="bg-[#e0e8f6] dark:bg-[#22324a]"
								selected={$page.url.pathname === '/about'}
							>
								<svelte:fragment slot="lead">
									<IconInfoOctagon />
								</svelte:fragment>
							</AppRailAnchor>
						</div>

						<!-- Settings Page Button -->
						<div use:popup={{ event: 'hover', target: 'settingsPageTooltip', placement: 'right' }}>
							<AppRailAnchor
								href="/settings"
								active="bg-[#e0e8f6] dark:bg-[#22324a]"
								selected={$page.url.pathname === '/settings'}
							>
								<svelte:fragment slot="lead">
									<IconSettings />
								</svelte:fragment>
							</AppRailAnchor>
						</div>

						<!-- Light Switch -->
						<AppRailAnchor hover="none">
							<svelte:fragment slot="lead">
								<LightSwitch width="w-10" height="h-5" />
							</svelte:fragment>
						</AppRailAnchor>
					</svelte:fragment>
				</AppRail>
			</div>
		{/if}
	</svelte:fragment>

	<!-- Databar -->
	<svelte:fragment slot="sidebarRight">
		<!-- Activate the Databar if Databar Button pressed -->
		{#if $rightBarActive && $page.url.pathname !== '/' && $page.url.pathname !== '/about'}
			<!-- Databar's Outer Container -->
			<div
				transition:slide={{ delay: 20, duration: 300, easing: quintInOut, axis: 'x' }}
				class="h-full flex ml-0.5 flex-col p-2"
			>
				<div class="grid h-full grid-rows-2 space-y-2">
					<!-- Text Area for Dataflow -->
					<div class="h-full p-2.5 rounded-sm space-y-4 bg-surface-200-700-token transition-all">
						<div class="flex flex-row mt-2 place-content-center">
							<h2 class="h4" style="font-family: Nevan">Options</h2>
						</div>
						<div class="grid grid-cols-1 h-full">
							<!-- Fine-tunning options -->
							<div class="grid-rows-2 auto-rows-fr h-full">
								<div class="h-full"><h1>Control Status</h1></div>
								<div class="h-full"><h2>Fine-tunning</h2></div>
							</div>
						</div>
					</div>
					<textarea
						class="textarea h-full shadow-xl overflow-y-auto border-2"
						placeholder="You can take your notes to here as you wish."
					/>
				</div>

				<!-- Restart Section -->
				<div class="flex flex-row w-full mt-4 place-items-center place-content-center gap-2">
					<!-- Reload AI Model Button -->
					<button
						on:click={() => {}}
						type="button"
						class="btn variant-filled w-full"
						use:popup={{ event: 'hover', target: 'raspberryTooltip', placement: 'top' }}
					>
						<IconRotate2 />
					</button>

					<!-- Restart AI Model Button-->
					<button
						on:click={() => {}}
						type="button"
						class="btn variant-filled w-full"
						use:popup={{ event: 'hover', target: 'pixhawkTooltip', placement: 'top' }}
					>
						<IconRotateClockwise2 />
					</button>
				</div>
			</div>
		{/if}
	</svelte:fragment>
	<slot />
</AppShell>
