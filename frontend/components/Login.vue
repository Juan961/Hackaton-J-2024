<template>
  <main class="main flex items-center justify-center min-h-screen">
    <div class="flex flex-col items-center">
      <h1 class="text-2xl font-medium mb-10">Login</h1>
      <form @submit.prevent="loginLogic" class="flex flex-col items-center gap-4">
        <input type="text" placeholder="Username" v-model="loginData.username" />
        <input type="password" placeholder="Password" v-model="loginData.password" />
        <button type="submit">{{ loginLoading ? 'Loading' : 'Login' }}</button>
      </form>
    </div>
  </main>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'

import { login } from '../actions/login'

const loginData = reactive({
  username: '',
  password: ''
})

const loginLoading = ref(false)
const loginError = ref(null)

const loginLogic = async () => {
  if (loginLoading.value) return
  if (!loginData.username || !loginData.password) {
    loginError.value = 'Please fill all the fields'
    return
  }

  loginLoading.value = true
  loginError.value = null

  const response = await login(loginData)

  loginLoading.value = false

  if (response === null) {
    loginError.value = 'Invalid credentials'

  } else {
    localStorage.setItem("accessToken", response)

    location.reload()
  }
}
</script>
