import { useState, useEffect } from 'react';

/**
 * Custom countdown timer hook for live IMD nowcast refresh
 */
export const useCountdown = (initialSeconds = 930) => {
  const [secondsLeft, setSecondsLeft] = useState(initialSeconds);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          return initialSeconds; // auto reset cycle
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [initialSeconds]);

  const reset = () => {
    setIsRefreshing(true);
    setTimeout(() => {
      setSecondsLeft(initialSeconds);
      setIsRefreshing(false);
    }, 400);
  };

  return { secondsLeft, reset, isRefreshing };
};
