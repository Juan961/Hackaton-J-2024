import axios from 'axios';

interface ILogin {
  username: string;
  password: string;
}

export const login = async (user: ILogin): Promise<string|null> => {
  try {
    const config = useRuntimeConfig();

    const url = config.public.apiUrl + '/login'

    const { data } = await axios.post(url, user);

    return data.accessToken;

  } catch (error) {
    console.error(error);

    return null;
  }
};
