<template>
  <main>
    <div class="login">
      <h1>Login</h1>
      <form @submit.prevent="loginLogic">
        <input type="text" placeholder="Username" />
        <input type="password" placeholder="Password" />
        <button type="submit">Login</button>
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

  if (response === null) {
    loginError.value = 'Invalid credentials'

  } else {
    localStorage.setItem("accessToken", response)

    location.reload()
  }
}
</script>
