import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/retrieval' },
    {
      path: '/retrieval',
      name: 'retrieval',
      component: () => import('@/pages/RetrievalPage.vue'),
      meta: { title: '检索问答' },
    },
    {
      path: '/evaluation',
      name: 'evaluation',
      component: () => import('@/pages/EvaluationPage.vue'),
      meta: { title: '评估数据集' },
    },
    {
      path: '/documents',
      name: 'documents',
      component: () => import('@/pages/DocumentPage.vue'),
      meta: { title: '文档管理' },
    },
    { path: '/:pathMatch(.*)*', redirect: '/retrieval' },
  ],
})

router.afterEach((to) => {
  const title = typeof to.meta.title === 'string' ? to.meta.title : ''
  document.title = title ? `${title} · ProductRAG` : 'ProductRAG'
})

export default router
