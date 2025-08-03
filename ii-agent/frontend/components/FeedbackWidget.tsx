// src/ii_agent/ui/components/FeedbackWidget.tsx
import React, { useState, useCallback, CSSProperties } from 'react';

// Types
interface FeedbackData {
  rating: number;
  feedback: string;
  suggestions: string;
  metadata: {
    userAgent: string;
    timestamp: string;
    executionId: string;
  };
}

interface StarProps {
  filled: boolean;
  onClick: () => void;
}

interface FeedbackWidgetProps {
  executionId: string;
  onFeedbackSubmit?: (feedback: FeedbackData) => void;
  apiEndpoint?: string;
  position?: 'fixed' | 'relative';
  theme?: 'light' | 'dark';
}

// Styles
const styles: Record<string, CSSProperties> = {
  widget: {
    padding: '20px',
    backgroundColor: '#f9f9f9',
    borderRadius: '8px',
    maxWidth: '500px',
    margin: '0 auto',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  },
  widgetDark: {
    backgroundColor: '#2a2a2a',
    color: '#ffffff',
  },
  title: {
    marginTop: 0,
    marginBottom: '20px',
    color: '#333',
  },
  titleDark: {
    color: '#ffffff',
  },
  ratingContainer: {
    marginBottom: '20px',
    textAlign: 'center' as const,
  },
  star: {
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    fontSize: '24px',
    padding: '0 2px',
    transition: 'color 0.2s',
  },
  ratingMessage: {
    textAlign: 'center' as const,
    color: '#666',
    fontSize: '14px',
    marginTop: '-10px',
    marginBottom: '20px',
  },
  ratingMessageDark: {
    color: '#aaa',
  },
  inputGroup: {
    marginBottom: '15px',
  },
  label: {
    display: 'block',
    marginBottom: '5px',
    color: '#555',
    fontSize: '14px',
    fontWeight: '500',
  },
  labelDark: {
    color: '#ddd',
  },
  textarea: {
    width: '100%',
    minHeight: '80px',
    padding: '10px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    fontSize: '14px',
    resize: 'vertical' as const,
    fontFamily: 'inherit',
    boxSizing: 'border-box' as const,
  },
  textareaDark: {
    backgroundColor: '#3a3a3a',
    color: '#ffffff',
    border: '1px solid #555',
  },
  submitButton: {
    width: '100%',
    padding: '12px',
    backgroundColor: '#007bff',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '16px',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  submitButtonDisabled: {
    backgroundColor: '#ccc',
    cursor: 'not-allowed',
  },
  successContainer: {
    textAlign: 'center' as const,
  },
  successIcon: {
    color: '#4CAF50',
    fontSize: '48px',
    marginBottom: '10px',
  },
  resetButton: {
    padding: '8px 20px',
    backgroundColor: '#007bff',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
  },
};

// Star component
const Star: React.FC<StarProps> = ({ filled, onClick }) => {
  const [isHovered, setIsHovered] = useState(false);

  const starStyle: CSSProperties = {
    ...styles.star,
    color: isHovered ? '#FFA500' : (filled ? '#FFD700' : '#D3D3D3'),
  };

  return (
    <button
      onClick={onClick}
      style={starStyle}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      aria-label={`Rate ${filled ? 'selected' : 'unselected'}`}
    >
      ★
    </button>
  );
};

// Main FeedbackWidget component
const FeedbackWidget: React.FC<FeedbackWidgetProps> = ({
  executionId,
  onFeedbackSubmit,
  apiEndpoint = '/api/feedback',
  position = 'relative',
  theme = 'light',
}) => {
  const [rating, setRating] = useState<number | null>(null);
  const [feedback, setFeedback] = useState('');
  const [suggestions, setSuggestions] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isDark = theme === 'dark';

  const getRatingMessage = (rating: number): string => {
    const messages: Record<number, string> = {
      1: "We're sorry to hear that 😔",
      2: "We'll work to improve 🤔",
      3: "Thanks for the feedback 👍",
      4: "Glad you had a good experience! 😊",
      5: "Awesome! Thanks for the great rating! 🎉",
    };
    return messages[rating] || '';
  };

  const submitFeedback = useCallback(async () => {
    if (!rating) {
      setError('Please provide a rating');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    const feedbackData: FeedbackData = {
      rating,
      feedback,
      suggestions,
      metadata: {
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
        executionId,
      },
    };

    try {
      // Make API call
      const response = await fetch(`${apiEndpoint}/${executionId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData),
      });

      if (!response.ok) {
        throw new Error('Failed to submit feedback');
      }

      // Call callback if provided
      if (onFeedbackSubmit) {
        onFeedbackSubmit(feedbackData);
      }

      setSubmitted(true);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      setError('Failed to submit feedback. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  }, [rating, feedback, suggestions, executionId, apiEndpoint, onFeedbackSubmit]);

  const resetForm = useCallback(() => {
    setRating(null);
    setFeedback('');
    setSuggestions('');
    setSubmitted(false);
    setError(null);
  }, []);

  const widgetStyle: CSSProperties = {
    ...styles.widget,
    ...(isDark ? styles.widgetDark : {}),
    position: position,
  };

  if (submitted) {
    return (
      <div style={widgetStyle}>
        <div style={styles.successContainer}>
          <div style={styles.successIcon}>✓</div>
          <h4 style={{ ...styles.title, ...(isDark ? styles.titleDark : {}) }}>
            Thank you for your feedback!
          </h4>
          <p style={{ color: isDark ? '#aaa' : '#666', marginBottom: '20px' }}>
            Your input helps us improve our service.
          </p>
          <button onClick={resetForm} style={styles.resetButton}>
            Submit Another Response
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={widgetStyle}>
      <h4 style={{ ...styles.title, ...(isDark ? styles.titleDark : {}) }}>
        How was your experience?
      </h4>

      {error && (
        <div style={{ color: '#f44336', marginBottom: '10px', fontSize: '14px' }}>
          {error}
        </div>
      )}

      <div style={styles.ratingContainer}>
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            filled={rating !== null && rating >= star}
            onClick={() => setRating(star)}
          />
        ))}
      </div>

      {rating && (
        <p style={{ ...styles.ratingMessage, ...(isDark ? styles.ratingMessageDark : {}) }}>
          {getRatingMessage(rating)}
        </p>
      )}

      <div style={styles.inputGroup}>
        <label style={{ ...styles.label, ...(isDark ? styles.labelDark : {}) }}>
          What went well or could be improved?
        </label>
        <textarea
          placeholder="Share your experience..."
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          style={{ ...styles.textarea, ...(isDark ? styles.textareaDark : {}) }}
          aria-label="Feedback details"
        />
      </div>

      <div style={styles.inputGroup}>
        <label style={{ ...styles.label, ...(isDark ? styles.labelDark : {}) }}>
          Any suggestions for improvement?
        </label>
        <textarea
          placeholder="How can we make this better?"
          value={suggestions}
          onChange={(e) => setSuggestions(e.target.value)}
          style={{ ...styles.textarea, ...(isDark ? styles.textareaDark : {}) }}
          aria-label="Improvement suggestions"
        />
      </div>

      <button
        onClick={submitFeedback}
        disabled={isSubmitting || !rating}
        style={{
          ...styles.submitButton,
          ...(isSubmitting || !rating ? styles.submitButtonDisabled : {}),
        }}
        aria-label="Submit feedback"
      >
        {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
      </button>
    </div>
  );
};

export default FeedbackWidget;