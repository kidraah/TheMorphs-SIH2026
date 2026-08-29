/**
 * Utility formatters for timestamps, risk levels, and meteorological units
 */

export const formatTime = (seconds) => {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

export const getRiskColor = (level) => {
  switch (level?.toLowerCase()) {
    case 'low':
      return { text: '#16A34A', bg: '#DCFCE7', border: '#86EFAC', label: 'Low' };
    case 'moderate':
      return { text: '#CA8A04', bg: '#FEF9C3', border: '#FDE047', label: 'Moderate' };
    case 'high':
      return { text: '#EA580C', bg: '#FFEDD5', border: '#FDBA74', label: 'High' };
    case 'very high':
    case 'veryhigh':
      return { text: '#DC2626', bg: '#FEE2E2', border: '#FCA5A5', label: 'Very High' };
    case 'extreme':
      return { text: '#7E22CE', bg: '#F3E8FF', border: '#D8B4FE', label: 'Extreme' };
    default:
      return { text: '#475569', bg: '#F1F5F9', border: '#CBD5E1', label: 'Normal' };
  }
};

export const getSeverityBadgeClass = (severity) => {
  switch (severity?.toLowerCase()) {
    case 'warning':
    case 'very high':
      return 'bg-red-50 text-red-700 border-red-300';
    case 'alert':
    case 'high':
      return 'bg-orange-50 text-orange-700 border-orange-300';
    case 'watch':
    case 'moderate':
      return 'bg-yellow-50 text-yellow-700 border-yellow-300';
    default:
      return 'bg-blue-50 text-blue-700 border-blue-300';
  }
};
