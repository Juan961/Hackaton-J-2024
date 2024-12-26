<template>
  <main>
    <form @submit.prevent="sendClassificationData">
      <input type="number" placeholder="Sunlight Hours" min="0" max="24" v-model="classificationData.sunlightHours">

      <input type="number" placeholder="Temperature" min="0" max="50" v-model="classificationData.temperature">

      <input type="number" placeholder="Humidity" min="0" max="100" v-model="classificationData.humidity">

      <select v-model="classificationData.soilType">
        <option value="" disabled>Soil_Type</option>
        <option value="sandy">sandy</option>
        <option value="loam">loam</option>
        <option value="clay">clay</option>
      </select>

      <select v-model="classificationData.waterFrequency">
        <option value="" disabled>Water_Frequency</option>
        <option value="weekly">weekly</option>
        <option value="bi-weekly">bi-weekly</option>
        <option value="daily">daily</option>
      </select>

      <select v-model="classificationData.fertilizerType">
        <option value="" disabled>Fertilizer_Type</option>
        <option value="organic">organic</option>
        <option value="chemical">chemical</option>
        <option value="none">none</option>
      </select>

      <p v-if="classificationDataError">{{ classificationDataError }}</p>
      <p v-if="classificationDataResponse !== null">{{ classificationDataResponse }}</p>

      <button type="submit">Analyze</button>
    </form>

    <form @submit.prevent="sendClassificationImage">
      <input type="file" @change="manageImage">

      <p v-if="classificationImageError">{{ classificationImageError }}</p>
      <p v-if="classificationImageResponse">{{ classificationImageResponse }}</p>

      <button type="submit">Analyze</button>
    </form>
  </main>
</template>


<script setup lang="ts">
import { ref, reactive } from 'vue'

import { predict } from '../actions/predict'

const classificationData = reactive({
  sunlightHours: 0,
  temperature: 0,
  humidity: 0,
  soilType: '',
  waterFrequency: '',
  fertilizerType: ''
})

interface IClassificationDataResponse {
  response: string
  growing: boolean
}

const classificationDataResponse = ref<IClassificationDataResponse|null>(null)
const classificationDataLoading = ref(false)
const classificationDataError = ref(null)


const classificationImage = ref(null)

interface IClassificationDataResponse {
  response: string
  plant: boolean
}

const classificationImageResponse = ref<IClassificationDataResponse|null>(null)
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

  classificationDataLoading.value = true


}

const sendClassificationImage = async () => {
  if (classificationImageLoading.value) return
  if (!classificationImage.value) {
    classificationImageError.value = 'Please upload an image'
    return
  }

  classificationImageError.value = null
  classificationImageResponse.value = null

  classificationImageLoading.value = true


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
