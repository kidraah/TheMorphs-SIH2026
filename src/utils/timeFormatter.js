/**
 * Centralized 24-Hour Indian Standard Time (IST) Formatter Utility
 * Timezone: Asia/Kolkata (UTC+05:30)
 * Format: 24-hour HH:mm (e.g., "20:02 IST", "14:30", "10:00 – 16:00 IST")
 */

/**
 * Returns current local time in 24-hour IST format (e.g., "20:02 IST")
 */
export const getCurrentISTTime = (includeIST = true) => {
  const now = new Date();
  const options = {
    timeZone: 'Asia/Kolkata',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  };
  const timeStr = new Intl.DateTimeFormat('en-GB', options).format(now);
  return includeIST ? `${timeStr} IST` : timeStr;
};

/**
 * Converts a 12-hour AM/PM string or time string into 24-hour format.
 * Examples:
 *   "10:20 AM" -> "10:20 IST"
 *   "01:30 PM" -> "13:30 IST"
 *   "10:00 AM" (with includeIST=false) -> "10:00"
 */
export const format24hIST = (timeString, includeIST = true) => {
  if (!timeString) return '';
  let str = timeString.trim();

  // Match 12-hour format like "10:20 AM", "01:30 PM", "10 AM", "01 PM"
  const match = str.match(/^(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)$/i);
  if (match) {
    let hours = parseInt(match[1], 10);
    const minutes = match[2] || '00';
    const period = match[3].toUpperCase();

    if (period === 'PM' && hours < 12) hours += 12;
    if (period === 'AM' && hours === 12) hours = 0;

    const formattedHours = hours.toString().padStart(2, '0');
    const formatted = `${formattedHours}:${minutes}`;
    return includeIST ? `${formatted} IST` : formatted;
  }

  // Handle strings like "12:30 PM IST" or "12:30 PM (IST)"
  str = str.replace(/\s*\(?IST\)?/i, '').trim();
  const match2 = str.match(/^(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)$/i);
  if (match2) {
    let hours = parseInt(match2[1], 10);
    const minutes = match2[2] || '00';
    const period = match2[3].toUpperCase();

    if (period === 'PM' && hours < 12) hours += 12;
    if (period === 'AM' && hours === 12) hours = 0;

    const formattedHours = hours.toString().padStart(2, '0');
    const formatted = `${formattedHours}:${minutes}`;
    return includeIST ? `${formatted} IST` : formatted;
  }

  return includeIST ? (str.endsWith('IST') ? str : `${str} IST`) : str.replace(/\s*IST/i, '').trim();
};

/**
 * Duration Countdown Formatter for HH:mm:ss (No AM/PM, No IST)
 * Example: 930 seconds -> "00:15:30"
 */
export const formatCountdownDuration = (seconds) => {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};
