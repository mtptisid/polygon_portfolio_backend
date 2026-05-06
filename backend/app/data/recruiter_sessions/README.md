# Recruiter Sessions Storage

This directory stores recruiter session data as JSON files.

## Structure

Each session is stored as a separate JSON file named `{session_id}.json`.

## File Format

```json
{
  "session_id": "uuid-string",
  "recruiter_info": {
    "name": "Recruiter Name",
    "email": "email@company.com",
    "company": "Company Name",
    "role": "Role Title",
    "additional_notes": "Optional notes"
  },
  "chat_history": [
    {
      "role": "user|assistant",
      "content": "Message content",
      "timestamp": "ISO8601 timestamp"
    }
  ],
  "analysis": {
    "conversation_summary": "Summary text",
    "key_topics_discussed": ["topic1", "topic2"],
    "role_fit_analysis": "Analysis text or null",
    "interest_level": "low|medium|high",
    "recommended_next_steps": ["step1", "step2"]
  },
  "metadata": {
    "start_time": "ISO8601 timestamp",
    "end_time": "ISO8601 timestamp",
    "session_duration_seconds": 1200,
    "total_messages": 12,
    "model_used": "gemini|groq"
  }
}
```

## Privacy

- Session files contain recruiter contact information
- Files are not committed to git (see .gitignore)
- Access requires admin authentication
- Files should be backed up regularly

## Maintenance

- Old sessions can be archived or deleted
- Recommended retention: 90 days
- Monitor disk usage as sessions accumulate
