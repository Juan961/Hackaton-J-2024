import axios from 'axios';

interface IDataClassificationImage {
  sunlightHours: number
  temperature: number
  humidity: number
  soilType: "sandy" | "loam" | "clay"
  waterFrequency: "daily" | "weekly" | "monthly"
  fertilizerType: number
}

interface IDataImage {
  base64: string;
}

interface IResponseClassification {
  growing: boolean
  response: string
}

interface IResponseImage {
  response: string
  plant: string
}

export const predict = async (accessToken: string, model:"image"|"classification", input:IDataClassificationImage|IDataImage): Promise<IResponseClassification|IResponseImage|null> => {
  try {
    const config = useRuntimeConfig();

    const url = config.public.apiUrl + '/predict/' + model;

    const { data } = await axios.post(url, input, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': accessToken
      }
    });

    return data;

  } catch (error) {
    console.error(error);

    return null;
  }
};
