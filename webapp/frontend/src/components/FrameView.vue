<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { api } from '../api'
import { store } from '../store'

const props = defineProps<{ cam: string; f0: number; view?: string; trail?: number; masks?: number; title?: string }>()
const emit = defineEmits<{ (e: 'change', f: number): void }>()

const frame = ref(props.f0)
const loading = ref(false)
const step = 5

watch(() => props.f0, v => { frame.value = v })
watch(() => props.cam, () => { frame.value = props.f0 })
watch(frame, (v, o) => { if (v !== o) emit('change', v) })

const url = computed(() => api.frameUrl(props.cam, frame.value, props.view || 'all', props.trail || 0, props.masks ?? 0) + `&_=${frame.value}`)
const camName = computed(() => store.camMap[props.cam]?.name || '')

function bump(d: number) { frame.value = Math.max(0, frame.value + d) }
function onSlider(e: Event) { frame.value = +(e.target as HTMLInputElement).value }
</script>

<template>
  <div class="fv panel col">
    <div class="panel-h">
      <span class="tag info mono">{{ cam }}</span>
      <span class="ellip">{{ title || camName }}</span>
      <div class="grow" />
      <span class="mono tiny dim">f={{ frame }}</span>
    </div>
    <div class="imgbox">
      <img :src="url" @loadstart="loading = true" @load="loading = false" :alt="`${cam} 第 ${frame} 帧`" />
      <div v-if="loading" class="spin"><span class="spinner" /></div>
    </div>
    <div class="ctrl row">
      <button class="icon-btn" @click="bump(-50)" title="后退 50 帧" aria-label="后退 50 帧">«</button>
      <button class="icon-btn" @click="bump(-step)" title="后退 5 帧" aria-label="后退 5 帧">−</button>
      <input type="range" :min="0" :max="store.camMap[cam]?.nframes || 15000" :value="frame"
             @input="onSlider" :aria-label="'帧号'" />
      <button class="icon-btn" @click="bump(step)" title="前进 5 帧" aria-label="前进 5 帧">+</button>
      <button class="icon-btn" @click="bump(50)" title="前进 50 帧" aria-label="前进 50 帧">»</button>
    </div>
  </div>
</template>

<style scoped>
.fv { min-width: 0; }
.imgbox {
  position: relative; background: var(--stage); aspect-ratio: 16 / 9;
  display: flex; align-items: center; justify-content: center;
}
.imgbox img { max-width: 100%; max-height: 100%; display: block; }
.spin { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; }
.ctrl { padding: var(--sp-2); gap: 6px; }
.ctrl input[type='range'] { flex: 1; accent-color: var(--accent); }
</style>
