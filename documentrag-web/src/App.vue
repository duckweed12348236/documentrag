<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import TaskDrawer from '@/components/TaskDrawer.vue'
import type { IconName } from '@/components/icons'
import { API_BASE_URL } from '@/plugins/request'
import { useTaskStore } from '@/stores/task'

interface NavItem {
  path: string
  label: string
  desc: string
  icon: IconName
}

const navItems: NavItem[] = [
  { path: '/retrieval', label: '检索问答', desc: '多轮检索与引用溯源', icon: 'chat' },
  { path: '/evaluation', label: '评估数据集', desc: '生成用例与指标评测', icon: 'dataset' },
  { path: '/documents', label: '文档管理', desc: 'PDF 上传与对象存储', icon: 'document' },
]

const fallbackNav: NavItem = navItems[0] ?? {
  path: '/retrieval',
  label: '检索问答',
  desc: '',
  icon: 'chat',
}

const route = useRoute()
const taskStore = useTaskStore()

const taskDrawerOpen = ref(false)
const navDrawerOpen = ref(false)

const currentNav = computed<NavItem>(
  () => navItems.find((item) => item.path === route.path) ?? fallbackNav,
)

const apiBase = API_BASE_URL

onMounted(() => {
  taskStore.start()
})

onBeforeUnmount(() => {
  taskStore.stop()
})
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <span class="brand-mark">
          <AppIcon name="document" :size="16" />
        </span>
        <span class="brand-text">
          <span class="brand-name">ProductRAG</span>
          <span class="brand-sub">文档知识工作台</span>
        </span>
      </div>

      <nav class="nav">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="nav-item"
          :class="{ 'is-active': route.path === item.path }"
        >
          <AppIcon :name="item.icon" :size="18" />
          <span class="nav-label">
            {{ item.label }}
            <span class="nav-desc">{{ item.desc }}</span>
          </span>
        </router-link>
      </nav>

      <div class="sidebar-foot">
        <span class="foot-label">API 服务</span>
        <span class="foot-url mono">{{ apiBase }}</span>
      </div>
    </aside>

    <div class="app-body">
      <header class="topbar">
        <button type="button" class="menu-btn" aria-label="打开导航" @click="navDrawerOpen = true">
          <AppIcon name="menu" :size="18" />
        </button>
        <div class="topbar-title">
          <span class="topbar-name">{{ currentNav.label }}</span>
          <span class="topbar-desc">{{ currentNav.desc }}</span>
        </div>
        <div class="topbar-actions">
          <el-button class="task-trigger" text @click="taskDrawerOpen = true">
            <AppIcon name="queue" :size="16" />
            <span class="task-trigger-label">任务队列</span>
            <span v-if="taskStore.activeCount" class="task-badge num">
              {{ taskStore.activeCount }}
            </span>
          </el-button>
        </div>
      </header>

      <main class="main">
        <div class="main-inner">
          <router-view v-slot="{ Component }">
            <transition name="page" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </div>
      </main>
    </div>

    <TaskDrawer v-model="taskDrawerOpen" />

    <el-drawer
      v-model="navDrawerOpen"
      direction="ltr"
      size="238px"
      :with-header="false"
      class="nav-drawer"
    >
      <div class="brand">
        <span class="brand-mark">
          <AppIcon name="document" :size="16" />
        </span>
        <span class="brand-text">
          <span class="brand-name">ProductRAG</span>
          <span class="brand-sub">文档知识工作台</span>
        </span>
      </div>
      <nav class="nav">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="nav-item"
          :class="{ 'is-active': route.path === item.path }"
          @click="navDrawerOpen = false"
        >
          <AppIcon :name="item.icon" :size="18" />
          <span class="nav-label">
            {{ item.label }}
            <span class="nav-desc">{{ item.desc }}</span>
          </span>
        </router-link>
      </nav>
    </el-drawer>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  height: 100vh;
  overflow: hidden;
  background: var(--surface);
}

.sidebar {
  display: flex;
  flex-direction: column;
  width: 238px;
  flex: none;
  background: var(--surface);
  border-right: 1px solid var(--hairline);
  padding: 20px 12px 16px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 8px;
}

.brand-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 7px;
  background: var(--accent);
  color: #fff;
}

.brand-text {
  display: flex;
  flex-direction: column;
  line-height: 1.3;
}

.brand-name {
  font-family: 'Aptos Display', Aptos, 'Microsoft YaHei', sans-serif;
  font-size: 15px;
  font-weight: 600;
  color: var(--ink);
}

.brand-sub {
  font-size: 11px;
  color: var(--muted);
}

.nav {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-top: 24px;
}

.nav-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 9px 10px;
  border-radius: 8px;
  color: var(--text);
  transition:
    background 160ms ease,
    color 160ms ease;
}

.nav-item:hover {
  background: var(--surface-soft);
}

.nav-item.is-active {
  background: var(--tint-lavender);
  color: var(--accent);
}

.nav-label {
  display: flex;
  flex-direction: column;
  line-height: 1.35;
  font-size: 13px;
  font-weight: 500;
}

.nav-desc {
  font-size: 11px;
  font-weight: 400;
  color: var(--muted);
}

.nav-item.is-active .nav-desc {
  color: var(--accent);
  opacity: 0.72;
}

.sidebar-foot {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-top: auto;
  padding: 12px 10px 0;
  border-top: 1px solid var(--hairline);
}

.foot-label {
  font-size: 11px;
  color: var(--muted);
}

.foot-url {
  font-size: 11px;
  color: var(--text);
  word-break: break-all;
}

.app-body {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
}

.topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  height: 56px;
  flex: none;
  padding: 0 24px;
  background: var(--canvas);
  border-bottom: 1px solid var(--hairline);
}

.menu-btn {
  display: none;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border: 1px solid var(--hairline);
  border-radius: 8px;
  background: var(--canvas);
  color: var(--text);
  cursor: pointer;
}

.topbar-title {
  display: flex;
  align-items: baseline;
  gap: 10px;
  min-width: 0;
}

.topbar-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--ink);
}

.topbar-desc {
  font-size: 12px;
  color: var(--muted);
}

.topbar-actions {
  margin-left: auto;
}

.task-trigger {
  color: var(--text);
}

.task-trigger-label {
  margin-left: 2px;
}

.task-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  height: 18px;
  margin-left: 6px;
  padding: 0 5px;
  border-radius: 6px;
  background: var(--tint-lavender);
  color: var(--accent);
  font-size: 11px;
  font-weight: 600;
}

.main {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

.main-inner {
  max-width: 1480px;
  margin: 0 auto;
}

@media (max-width: 900px) {
  .sidebar {
    display: none;
  }

  .menu-btn {
    display: inline-flex;
  }

  .topbar {
    padding: 0 16px;
  }

  .topbar-desc {
    display: none;
  }

  .main {
    padding: 16px;
  }
}
</style>

<style>
.nav-drawer .el-drawer__body {
  display: flex;
  flex-direction: column;
  padding: 20px 12px 16px;
  background: var(--surface);
}
</style>
