# Profile Data for RAG Embedding

This directory contains structured profile data for Siddharamayya Mathapati, organized by category for efficient vector embedding and retrieval.

## File Structure

```
data/
├── about.json              # Personal info, bio, availability
├── skills.json             # Technical skills with proficiency levels
├── experience.json         # Work experience with detailed descriptions
├── projects.json           # Portfolio projects with technologies
├── education.json          # Academic qualifications
├── certifications.json     # Professional certifications
├── contact.json            # Contact information and social links
└── README.md              # This file
```

## Data Organization

Each JSON file is structured for optimal chunking and embedding:

- **about.json**: Single object with personal details
- **skills.json**: Array of skill objects with categories
- **experience.json**: Array of work experiences with key technologies
- **projects.json**: Array of projects with URLs and tech stacks
- **education.json**: Array of degrees with dates
- **certifications.json**: Array of certifications with providers
- **contact.json**: Single object with all contact methods

## Usage for RAG

### Chunking Strategy

1. **About**: Embed as single chunk (high-level overview)
2. **Skills**: Embed each skill individually with category metadata
3. **Experience**: Embed each job separately, optionally split by key achievements
4. **Projects**: Embed each project individually with technology tags
5. **Education**: Embed each degree separately
6. **Certifications**: Embed as group or individually

### Metadata Fields

Each chunk should include:
- `category`: Data type (about, skill, experience, project, education, certification)
- `subcategory`: Specific area (e.g., "AI/ML", "DevOps", "IoT")
- `technologies`: List of relevant tech (for filtering)
- `date_range`: For experience/education (for temporal queries)

### Embedding Model Recommendations

- **HuggingFace**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, free)
- **Groq**: Use Groq embeddings API (if available)
- **OpenAI**: `text-embedding-3-small` (1536 dimensions, paid)

### PostgreSQL pgvector Schema

```sql
CREATE TABLE profile_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),  -- adjust dimension based on model
    category VARCHAR(50),
    subcategory VARCHAR(100),
    technologies TEXT[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON profile_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON profile_embeddings (category);
CREATE INDEX ON profile_embeddings USING GIN (technologies);
```

## Next Steps

1. Create embedding script (`scripts/embed_profile.py`)
2. Set up PostgreSQL with pgvector on Render
3. Ingest data into vector store
4. Update `ai_chat.py` to use RAG retrieval instead of few-shot prompts
5. Test with sample queries

## Sample Queries for Testing

- "What is Siddharamayya's experience with LangChain?"
- "Tell me about his projects using IoT"
- "What are his AI/ML skills?"
- "When did he work at Capgemini?"
- "Show me projects related to Ansible"
