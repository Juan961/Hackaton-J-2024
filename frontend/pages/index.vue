<template>
  <div>
    <Login v-if="!logged" />
    <Platform v-else />
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'

const logged = ref(false)

onMounted(() => {
  const token = localStorage.getItem('token')
  if (token) {
    logged.value = true
  }
})
</script>
