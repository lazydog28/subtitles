import tailwindcss from "@tailwindcss/vite";

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  vite: {
    plugins: [tailwindcss()],
    build:{
      target:"esnext" // 启用现代ES特性提升GPU加速效果
    },
  },
  css: ["~/assets/app.css"],
  ssr:false,
  experimental:{
    payloadExtraction:false // 避免静态提取干扰硬件加速
  }
})
