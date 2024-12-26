<template>
  <main class="main flex flex-col items-center justify-center min-h-screen gap-4 my-10">
    <form @submit.prevent="sendClassificationData" class="flex flex-col items-center w-full max-w-sm gap-3">
      <h2 class="text-2xl font-medium mb-2">Classification</h2>

      <label for="sunlightHours" class="w-full text-left">Sunlight Hours</label>
      <input id="sunlightHours" type="number" placeholder="Sunlight Hours" min="0" max="24" class="w-full border px-4 py-2 rounded" v-model="classificationData.sunlightHours">

      <label for="temperature" class="w-full text-left">Temperature</label>
      <input id="temperature" type="number" placeholder="Temperature" min="0" max="50" class="w-full border px-4 py-2 rounded" v-model="classificationData.temperature">

      <label for="humidity" class="w-full text-left">Humidity</label>
      <input id="humidity" type="number" placeholder="Humidity" min="0" max="100" class="w-full border px-4 py-2 rounded" v-model="classificationData.humidity">

      <label for="soilType" class="w-full text-left">Soil Type</label>
      <select id="soilType" v-model="classificationData.soilType" class="w-full">
        <option value="sandy">sandy</option>
        <option value="loam">loam</option>
        <option value="clay">clay</option>
      </select>

      <label for="waterFrequency" class="w-full text-left">Water Frequency</label>
      <select id="waterFrequency" v-model="classificationData.waterFrequency" class="w-full">
        <option value="weekly">weekly</option>
        <option value="bi-weekly">bi-weekly</option>
        <option value="daily">daily</option>
      </select>

      <label for="fertilizerType" class="w-full text-left">Fertilizer Type</label>
      <select id="fertilizerType" v-model="classificationData.fertilizerType" class="w-full">
        <option value="organic">organic</option>
        <option value="chemical">chemical</option>
        <option value="none">none</option>
      </select>

      <p class="w-full" v-if="classificationDataError">{{ classificationDataError }}</p>
      <p class="w-full" v-if="classificationDataResponse !== null">{{ classificationDataResponse }}</p>

      <button class="w-full bg-green-500 rounded text-white" type="submit">Analyze</button>
    </form>

    <form @submit.prevent="sendClassificationImage" class="flex flex-col items-center w-full max-w-sm gap-3">
      <h2 class="text-2xl font-medium mb-2">Image Classification</h2>

      <label for="imageUpload" class="w-full text-left">Upload Image</label>
      <input id="imageUpload" type="file" @change="manageImage" class="w-full">

      <p class="w-full" v-if="classificationImageError">{{ classificationImageError }}</p>
      <p class="w-full" v-if="classificationImageResponse">{{ classificationImageResponse }}</p>

      <button class="w-full bg-green-500 rounded text-white" type="submit">Analyze</button>
    </form>
  </main>
</template>


<script setup lang="ts">
import { ref, reactive } from 'vue'

import { predict, IResponseClassification, IResponseImage } from '../actions/predict'

const classificationData = reactive({
  sunlightHours: 0,
  temperature: 0,
  humidity: 0,
  soilType: 'sandy',
  waterFrequency: 'weekly',
  fertilizerType: 'organic'
})

const classificationDataResponse = ref<IResponseClassification|null>(null)
const classificationDataLoading = ref(false)
const classificationDataError = ref(null)


const classificationImage = ref(null)

const classificationImageResponse = ref<IResponseImage|null>(null)
const classificationImageLoading = ref(false)
const classificationImageError = ref(null)


const sendClassificationData = async () => {
  if (classificationDataLoading.value) return
  if (!classificationData.sunlightHours || !classificationData.temperature || !classificationData.humidity || !classificationData.soilType || !classificationData.waterFrequency || !classificationData.fertilizerType) {
    classificationDataError.value = 'Please fill all the fields'
    return
  }

  classificationDataError.value = null
  classificationDataResponse.value = null

  const accessToken = localStorage.getItem('accessToken')

  if (!accessToken) {
    location.reload()
    return
  }

  classificationDataLoading.value = true

  const result = await predict(accessToken, "classification", classificationData)

  classificationDataLoading.value = false

  if (result === null) {
    classificationDataError.value = 'Invalid data'
  } else if ('growing' in result) {
    classificationDataResponse.value = result as IResponseClassification
  } else {
    classificationDataError.value = 'Unexpected response type'
  }
}

const sendClassificationImage = async () => {
  if (classificationImageLoading.value) return
  if (!classificationImage.value) {
    classificationImageError.value = 'Please upload an image'
    return
  }

  classificationImageError.value = null
  classificationImageResponse.value = null

  const accessToken = localStorage.getItem('accessToken')

  if (!accessToken) {
    location.reload()
    return
  }

  classificationImageLoading.value = true

  const result = await predict(accessToken, "classification", classificationImage.value)

  classificationImageLoading.value = false

  if (result === null) {
    classificationImageError.value = 'Invalid image'
  } else {
    if ('response' in result && 'plant' in result) {
      classificationImageResponse.value = result as IResponseImage
    } else {
      classificationImageError.value = 'Unexpected response type'
    }
  }
}

const manageImage = (e: any) => {
  classificationImage.value = null

  if (e.target.files[0].size > 1024 * 1024) {
    classificationImageError.value = 'The image size must be less than 1MB'
    return
  }

  classificationImage.value = e.target.files[0]
}
</script>
