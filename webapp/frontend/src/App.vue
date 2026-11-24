<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import { store, loadAll, tick } from './store'
import { theme, initTheme, toggleTheme } from './theme'

const route = useRoute()
const health = ref<any>(null)
let raf = 0

onMounted(async () => {
  initTheme()
  try {
    const r = await fetch('/api/health'); health.value = await r.json()
  } catch { health.value = { ok: false } }
  await loadAll()
  const loop = () => { tick(); raf = requestAnimationFrame(loop) }
  raf = requestAnimationFrame(loop)
})
onUnmounted(() => cancelAnimationFrame(raf))

const navItems = [
  { to: '/ops', label: '总览', short: '总览', icon: 'ops' },
  { to: '/', label: '可视化', short: '可视化', icon: 'view' },
  { to: '/annotate', label: '素材标注', short: '标注', icon: 'annotate' },
  { to: '/reid', label: 'ReID · 语义', short: 'ReID', icon: 'search' },
  { to: '/selector', label: '身份选择器', short: '选择器', icon: 'selector' },
  { to: '/auto-face', label: '自动人脸', short: '人脸', icon: 'autoface' },
  { to: '/cluster', label: '聚类工作台', short: '聚类', icon: 'cluster' },
  { to: '/truth', label: '真值标注', short: '真值', icon: 'truth' },
  { to: '/e21-label', label: 'E21 标簇', short: 'E21', icon: 'truth' },
  { to: '/faces', label: '人脸库', short: '人脸', icon: 'faces', hideSm: true },
  { to: '/runs', label: '算法运行', short: '运行', icon: 'runs', hideSm: true },
  { to: '/train', label: '训练过程', short: '训练', icon: 'train', hideSm: true },
  { to: '/jobs', label: '任务派发', short: '派发', icon: 'jobs' },
  { to: '/workbench', label: '实验卡', short: '实验', icon: 'workbench' },
  { to: '/config', label: '算法配置', short: '配置', icon: 'config' }
]
const smNav = computed(() => navItems.filter(n => !(n as any).hideSm))
const supraOk = computed(() => health.value?.supervision)
const reidOk = computed(() => health.value?.reid)
const current = computed(() => navItems.find(n => n.to === route.path)?.label || '')
</script>

<template>
  <div class="app">
    <header class="bar">
      <RouterLink to="/" class="brand" aria-label="human-count 首页">
        <svg class="mark" viewBox="0 0 32 32" aria-hidden="true">
          <circle cx="16" cy="16" r="5.2" />
          <circle cx="16" cy="16" r="12.2" opacity=".38" />
          <g class="nodes">
            <circle cx="16" cy="3.8" r="2.2" /><circle cx="26.6" cy="9.9" r="2.2" />
            <circle cx="26.6" cy="22.1" r="2.2" /><circle cx="16" cy="28.2" r="2.2" />
            <circle cx="5.4" cy="22.1" r="2.2" /><circle cx="5.4" cy="9.9" r="2.2" />
          </g>
        </svg>
        <span class="word">
          <b>human&#8209;count</b>
          <i class="label">多机位追踪控制台</i>
        </span>
      </RouterLink>

      <nav class="nav hide-sm" aria-label="主导航">
        <RouterLink v-for="n in navItems" :key="n.to" :to="n.to" class="nav-item"
                    :aria-current="route.path === n.to ? 'page' : undefined">
          {{ n.label }}
        </RouterLink>
      </nav>

      <span class="only-sm page-title ellip">{{ current }}</span>

      <div class="grow" />

      <div class="status hide-sm">
        <span class="tag" :class="health?.ok ? 'ok' : 'bad'">
          <i class="dot" /> API {{ health?.ok ? '在线' : '离线' }}
        </span>
        <span class="tag" :class="supraOk ? 'info' : 'warn'">
          <i class="dot" /> supervision {{ supraOk ? '就绪' : '未启用' }}
        </span>
        <span class="tag" :class="reidOk ? 'info' : 'warn'" title="全局 ReID 库轨迹数">
          <i class="dot" /> ReID {{ health?.reid_n ?? 0 }}
        </span>
        <span v-if="store.ready" class="tag mono">窗口 {{ store.session.window_sec }}s</span>
      </div>

      <button class="icon-btn" @click="toggleTheme()"
              :title="theme === 'dark' ? '切换到浅色主题' : '切换到深色主题'"
              :aria-label="theme === 'dark' ? '切换到浅色主题' : '切换到深色主题'">
        <svg viewBox="0 0 20 20" width="15" height="15" aria-hidden="true">
          <g v-if="theme === 'dark'" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
            <circle cx="10" cy="10" r="3.6" />
            <path d="M10 2v2M10 16v2M2 10h2M16 10h2M4.3 4.3l1.4 1.4M14.3 14.3l1.4 1.4M15.7 4.3l-1.4 1.4M5.7 14.3l-1.4 1.4" />
          </g>
          <path v-else fill="currentColor" d="M16.5 12.4A7 7 0 0 1 7.6 3.5a7 7 0 1 0 8.9 8.9Z" />
        </svg>
      </button>
    </header>

    <div v-if="store.error" class="fallback">
      <div class="panel" style="max-width:520px">
        <div class="panel-h">数据加载失败</div>
        <div class="panel-b col" style="gap:8px">
          <p class="small">{{ store.error }}</p>
          <p class="tiny dim">请确认后端已启动，且 <span class="mono">webapp/backend/data</span> 数据包存在。</p>
          <button class="primary" @click="() => location.reload()">重新加载</button>
        </div>
      </div>
    </div>
    <div v-else-if="!store.ready" class="fallback">
      <div class="boot">
        <span class="spinner" />
        <div class="col" style="gap:2px">
          <b class="small">正在加载多机位数据…</b>
          <span class="tiny dim">6 路轨迹 · 实体图 · 接力链 · 全局 ReID 库</span>
        </div>
      </div>
    </div>
    <main v-else id="main" class="main" tabindex="-1">
      <RouterView />
    </main>

    <nav class="tabbar only-sm" aria-label="主导航">
      <RouterLink v-for="n in smNav" :key="n.to" :to="n.to" class="tab"
                  :aria-current="route.path === n.to ? 'page' : undefined">
        <svg viewBox="0 0 24 24" width="19" height="19" aria-hidden="true" fill="none"
             stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
          <template v-if="n.icon === 'ops'">
            <path d="M3.5 15.5a8.5 8.5 0 1 1 17 0" />
            <path d="M12 15.5 15.8 10" />
            <circle cx="12" cy="15.5" r="1.3" />
            <path d="M5.6 12.2h1.6M16.8 12.2h1.6M8.4 7.4l1.1 1.1M15.6 7.4l-1.1 1.1" />
          </template>
          <template v-else-if="n.icon === 'jobs'">
            <rect x="3.5" y="4.5" width="6.5" height="15" rx="1.6" />
            <rect x="13.5" y="4.5" width="6.5" height="9" rx="1.6" />
            <path d="M6.8 8.2h.01M6.8 11.5h.01M16.8 8.2h.01" />
          </template>
          <template v-else-if="n.icon === 'train'">
            <path d="M3.5 17.5 8 12l3.5 3 5-7.5" />
            <path d="M3.5 20.5h17" />
            <circle cx="8" cy="12" r="1.1" /><circle cx="11.5" cy="15" r="1.1" />
            <circle cx="16.5" cy="7.5" r="1.1" />
          </template>
          <template v-else-if="n.icon === 'runs'">
            <path d="M4 6.5h9M4 12h13M4 17.5h6" />
            <circle cx="18" cy="6.5" r="2" /><circle cx="8" cy="17.5" r="2" />
          </template>
          <template v-else-if="n.icon === 'view'">
            <rect x="3" y="4" width="7.5" height="6.5" rx="1.4" /><rect x="13.5" y="4" width="7.5" height="6.5" rx="1.4" />
            <rect x="3" y="13.5" width="7.5" height="6.5" rx="1.4" /><rect x="13.5" y="13.5" width="7.5" height="6.5" rx="1.4" />
          </template>
          <template v-else-if="n.icon === 'annotate'">
            <path d="M4 20l3.2-.7L19 7.5a2 2 0 0 0-2.8-2.8L4.4 16.5 4 20Z" /><path d="M14.5 6.5l3 3" />
          </template>
          <template v-else-if="n.icon === 'search'">
            <circle cx="10.5" cy="10.5" r="6.2" /><path d="M15.2 15.2 20 20" />
          </template>
          <template v-else-if="n.icon === 'autoface'">
            <circle cx="12" cy="12" r="6.4" />
            <circle cx="9.4" cy="10.6" r="0.9" /><circle cx="14.6" cy="10.6" r="0.9" />
            <path d="M9.1 14.6c1.7 1.5 4.1 1.5 5.8 0" />
            <path d="M3.2 8.2V5.6h2.6M18.2 8.2V5.6h-2.6M3.2 15.8v2.6h2.6M18.2 15.8v2.6h-2.6" />
          </template>
          <template v-else-if="n.icon === 'cluster'">
            <circle cx="8" cy="8" r="3.1" /><circle cx="16.5" cy="10" r="2.4" /><circle cx="10.5" cy="17" r="2.6" />
            <path d="M10.6 9.4l3.6.4M9.4 10.8l.6 3.6M14.9 12.1l-2.6 3.2" />
          </template>
          <template v-else-if="n.icon === 'truth'">
            <path d="M4.5 5.2h9.2l5.8 5.8v8H4.5z" /><path d="M8.2 13.6l2.6 2.6 5-5" />
          </template>
          <template v-else-if="n.icon === 'workbench'">
            <rect x="3.6" y="5" width="12.4" height="15" rx="1.8" />
            <path d="M7 9.2h5.6M7 13h3.4" />
            <rect x="17.4" y="8.6" width="3" height="3" rx=".7" />
            <rect x="17.4" y="14.4" width="3" height="3" rx=".7" />
            <path d="M18.9 3.4v2.4" />
          </template>
          <template v-else>
            <circle cx="12" cy="12" r="2.8" />
            <path d="M12 3.6v3M12 17.4v3M3.6 12h3M17.4 12h3M6.1 6.1l2.1 2.1M15.8 15.8l2.1 2.1M17.9 6.1l-2.1 2.1M8.2 15.8l-2.1 2.1" />
          </template>
        </svg>
        <span>{{ n.short }}</span>
      </RouterLink>
    </nav>
  </div>
</template>

<style scoped>
.app { display: flex; flex-direction: column; height: 100%; }

/* -------- 顶栏 -------- */
.bar {
  display: flex; align-items: center; gap: var(--sp-3);
  min-height: var(--bar-h); flex: none;
  padding: 0 var(--sp-3);
  padding-top: env(safe-area-inset-top);
  background: var(--surface-1);
  border-bottom: 1px solid var(--line);
  position: relative; z-index: var(--z-bar);
}
.brand { display: flex; align-items: center; gap: var(--sp-2); color: var(--fg); flex: none; }
.brand:hover { text-decoration: none; }
.mark { width: 26px; height: 26px; flex: none; color: var(--accent); }
.mark circle { fill: none; stroke: currentColor; stroke-width: 1.6; }
.mark > circle:first-child { fill: currentColor; stroke: none; }
.mark .nodes circle { fill: currentColor; stroke: none; opacity: .75; }
.word { display: flex; flex-direction: column; line-height: 1.15; }
.word b { font-size: var(--text-base); font-weight: 650; letter-spacing: -0.01em; }
.word i { font-style: normal; }

.nav {
  display: flex; gap: var(--sp-1); min-width: 0;
  overflow-x: auto; scrollbar-width: none;
}
.nav::-webkit-scrollbar { display: none; }
.nav-item {
  padding: 5px var(--sp-3); border-radius: var(--r-sm); color: var(--fg-2);
  font-size: var(--text-sm); font-weight: 500; border: 1px solid transparent;
  white-space: nowrap;
  transition: all var(--dur-instant) var(--ease-out-quart);
}
@media (hover: hover) { .nav-item:hover { background: var(--surface-2); color: var(--fg); text-decoration: none; } }
.nav-item[aria-current='page'] {
  background: var(--accent-soft); color: var(--accent);
  border-color: var(--accent-line); font-weight: var(--weight-label);
}
.status { display: flex; gap: 6px; align-items: center; }
.page-title { font-weight: var(--weight-label); color: var(--fg-2); max-width: 42vw; }

/* -------- 主体 -------- */
.main { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.main > * { flex: 1; min-height: 0; }

.fallback { flex: 1; display: flex; align-items: center; justify-content: center; padding: var(--sp-5); }
.boot { display: flex; align-items: center; gap: var(--sp-3); color: var(--fg-2); }

/* -------- 移动端底部标签栏 -------- */
.tabbar {
  flex: none; display: none; align-items: stretch;
  background: var(--surface-1); border-top: 1px solid var(--line);
  padding-bottom: env(safe-area-inset-bottom);
}
@media (max-width: 767px) { .tabbar { display: flex; } }
.tab {
  position: relative; flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 2px; padding: 6px 2px; min-height: 52px;
  color: var(--fg-3); font-size: var(--text-2xs); font-weight: var(--weight-label);
  transition: color var(--dur-instant) var(--ease-out-quart);
}
.tab:hover { text-decoration: none; }
.tab[aria-current='page'] { color: var(--accent); }
.tab[aria-current='page']::after {
  content: ''; position: absolute; top: 0; width: 26px; height: 2px;
  background: var(--accent); border-radius: 0 0 2px 2px;
}

/* 中宽：压缩低频元素，保证导航永不换行 */
@media (max-width: 1279px) { .status .tag:nth-child(n+4) { display: none; } }
@media (max-width: 1099px) { .status .tag:nth-child(n+3) { display: none; } }
@media (max-width: 1023px) { .word i { display: none; } .nav-item { padding: 5px var(--sp-2); } }
@media (max-width: 899px) { .status { display: none; } }
@media (max-width: 767px) {
  .bar { gap: var(--sp-2); }
  .word i { display: none; }
  .nav { display: none; }
  .status { display: none; }
}
</style>
