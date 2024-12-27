<template>
  <main class="main flex flex-col items-center justify-center min-h-screen gap-4 my-10">
    <form @submit.prevent="sendClassificationData" class="flex flex-col items-center w-full max-w-sm gap-3">
      <h2 class="text-2xl font-medium mb-2">Classification</h2>

      <label for="sunlightHours" class="w-full text-left">Sunlight Hours [0h-24h]</label>
      <input id="sunlightHours" type="number" placeholder="Sunlight Hours" min="0" max="24" class="w-full border px-4 py-2 rounded" v-model="classificationData.Sunlight_Hours">

      <label for="temperature" class="w-full text-left">Temperature [0°C-50°C]</label>
      <input id="temperature" type="number" placeholder="Temperature" min="0" max="50" class="w-full border px-4 py-2 rounded" v-model="classificationData.Temperature">

      <label for="humidity" class="w-full text-left">Humidity [0%-100%]</label>
      <input id="humidity" type="number" placeholder="Humidity" min="0" max="100" class="w-full border px-4 py-2 rounded" v-model="classificationData.Humidity">

      <label for="soilType" class="w-full text-left">Soil Type</label>
      <select id="soilType" v-model="classificationData.Soil_Type" class="w-full border px-4 py-2 rounded">
        <option value="sandy">Sandy</option>
        <option value="loam">Loam</option>
        <option value="clay">Clay</option>
      </select>

      <label for="waterFrequency" class="w-full text-left">Water Frequency</label>
      <select id="waterFrequency" v-model="classificationData.Water_Frequency" class="w-full border px-4 py-2 rounded">
        <option value="weekly">Weekly</option>
        <option value="bi-weekly">Bi Weekly</option>
        <option value="daily">Daily</option>
      </select>

      <label for="fertilizerType" class="w-full text-left">Fertilizer Type</label>
      <select id="fertilizerType" v-model="classificationData.Fertilizer_Type" class="w-full border px-4 py-2 rounded">
        <option value="organic">Organic</option>
        <option value="chemical">Chemical</option>
        <option value="none">None</option>
      </select>

      <p class="w-full" v-if="classificationDataError">{{ classificationDataError }}</p>
      <p class="w-full" v-if="classificationDataResponse">{{ classificationDataResponse.growing ? 'Your plant should be growing, well done' : 'Be careful your plant may not be growing as excepted' }}</p>
      <p class="w-full" v-if="classificationDataResponse">{{ classificationDataResponse.response }}</p>

      <button class="w-full bg-green-500 rounded text-white py-2 text-lg" type="submit">{{ classificationDataLoading ? 'Loading' : 'Analyze' }}</button>
    </form>

    <form @submit.prevent="sendClassificationImage" class="flex flex-col items-center w-full max-w-sm gap-3">
      <h2 class="text-2xl font-medium mb-2">Image Classification</h2>

      <label for="imageUpload" class="w-full text-left">Upload Image</label>
      <input id="imageUpload" type="file" @change="manageImage" class="w-full" accept="image/*">

      <p class="w-full" v-if="classificationImageError">{{ classificationImageError }}</p>
      <p class="w-full" v-if="classificationImageResponse">Your plant looks like: {{ classificationImageResponse.plant }}</p>
      <p class="w-full" v-if="classificationImageResponse">{{ classificationImageResponse.response }}</p>

      <button class="w-full bg-green-500 rounded text-white py-2 text-lg" type="submit">{{ classificationImageLoading ? 'Loading' : 'Analyze' }}</button>
    </form>
  </main>
</template>


<script setup lang="ts">
import { ref, reactive } from 'vue'

import { predict, IResponseClassification, IResponseImage } from '../actions/predict'

const classificationData = reactive({
  "Sunlight_Hours": 0,
  "Temperature": 0,
  "Humidity": 0,
  "Soil_Type": 'sandy',
  "Water_Frequency": 'weekly',
  "Fertilizer_Type": 'organic'
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
  if (!classificationData.Soil_Type || !classificationData.Water_Frequency || !classificationData.Fertilizer_Type) {
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

  const reader = new FileReader()
  reader.readAsDataURL(classificationImage.value)
  reader.onload = async () => {
    const base64Image = reader.result as string

    classificationImageLoading.value = true

    const result = await predict(accessToken, "image", { image: base64Image })

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
  reader.onerror = () => {
    classificationImageError.value = 'Error reading image'
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
