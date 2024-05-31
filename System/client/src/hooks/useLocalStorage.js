import { useState } from 'react';

function useLocalStorage(key, initialValue) {
  const [storedValue, setStoredValue] = useState(
    () => {
      try {
        const item = window.localStorage.getItem(key);
        return item ? JSON.parse(item) : initialValue;
      } catch (error) {
        console.log(error);
        return initialValue;
      }
    }
  );

  const setValue = (value) => {
    try {
      setStoredValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.log(error);
    }
  };

  return [storedValue, setValue];
}

// const [isDarkTheme, setDarkTheme] = useLocalStorage('darkTheme', true)
// const toggleTheme = () => {
//   setDarkTheme((prevValue: boolean) => !prevValue)
// }
export function useLocalStorageInit(key, initialValue) {
  const readValue = useCallback(() => {
    if (typeof window === 'undefined') {
      return initialValue
    }

    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : initialValue;
    } catch(error) {
      console.log('读取localStorage错误', error);
      return initialValue;
    }
  }, [key, initialValue])

  const [storedValue, setStoredValue] = useState(readValue)

  const setValue = value => {
    try {
      setStoredValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch(error) {
      console.log('存储localStorage错误', error);
    }
  }

  return [storedValue, setValue];
}