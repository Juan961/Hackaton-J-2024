<template>
  <Login v-if="!logged" />
  <Platform v-else />
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'

const logged = ref(false)

useHead({
  title: logged ? 'Platform' : 'Login',
  script: [
    { src: "https://cdn.tailwindcss.com" }
  ]
})

onMounted(() => {
  const token = localStorage.getItem('accessToken')
  if (token) {
    logged.value = true
  }
})
</script>
