#!/usr/bin/env python3
"""
Profile Data Ingestion Script

This script loads JSON files from backend/app/data/ and populates the vector
database with embeddings. It implements category-specific chunking strategies
and supports CLI arguments for flexible ingestion.

Usage:
    python backend/app/scripts/ingest_profile_data.py
    python backend/app/scripts/ingest_profile_data.py --dry-run
    python backend/app/scripts/ingest_profile_data.py --category skills
    python backend/app/scripts/ingest_profile_data.py --data-dir /path/to/data
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from myapp.services.embedding import EmbeddingService
from myapp.services.vector_store import VectorStoreManager, EmbeddingData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Data class representing a chunk of content to be embedded."""
    content: str
    category: str
    subcategory: Optional[str] = None
    technologies: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ProfileDataIngestion:
    """
    Handles the ingestion of profile data into the vector database.
    
    Features:
    - JSON file loading and validation
    - Category-specific chunking strategies
    - Batch embedding generation
    - Progress logging
    - Dry-run mode for preview
    """
    
    # Expected JSON files and their categories
    DATA_FILES = {
        'about.json': 'about',
        'skills.json': 'skills',
        'experience.json': 'experience',
        'projects.json': 'projects',
        'education.json': 'education',
        'certifications.json': 'certifications',
        'contact.json': 'contact'
    }
    
    def __init__(
        self,
        data_dir: str,
        embedding_service: EmbeddingService,
        vector_store: Optional[VectorStoreManager] = None
    ):
        """
        Initialize the ingestion handler.
        
        Args:
            data_dir: Directory containing JSON data files
            embedding_service: Service for generating embeddings
            vector_store: Vector store manager (None for dry-run)
        """
        self.data_dir = Path(data_dir)
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def load_json_files(self, category_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Load all JSON files from the data directory.
        
        Args:
            category_filter: Optional category to load (e.g., "skills")
            
        Returns:
            Dictionary mapping category names to loaded data
            
        Raises:
            FileNotFoundError: If required files are missing
            json.JSONDecodeError: If JSON is malformed
        """
        loaded_data = {}
        
        for filename, category in self.DATA_FILES.items():
            # Skip if category filter is set and doesn't match
            if category_filter and category != category_filter:
                continue
            
            file_path = self.data_dir / filename
            
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    loaded_data[category] = data
                    self.logger.info(f"Loaded {filename} ({category})")
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse {filename}: {e}")
                # Continue processing other files
                continue
            except Exception as e:
                self.logger.error(f"Error loading {filename}: {e}")
                continue
        
        if not loaded_data:
            raise ValueError("No data files loaded successfully")
        
        return loaded_data
    
    def chunk_about(self, data: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk 'about' data into a single chunk.
        
        Args:
            data: About data dictionary
            
        Returns:
            List containing single chunk
        """
        # Create comprehensive about content
        content_parts = [
            f"Name: {data.get('full_name', 'N/A')}",
            f"Professional Summary: {data.get('professional_summary', data.get('description', 'N/A'))}",
            f"Current Role: {data.get('current_role', 'N/A')} at {data.get('current_company', 'N/A')}",
            f"Experience: {data.get('total_experience_years', 'N/A')} years",
            f"Location: {data.get('location', 'N/A')}",
            f"Languages: {', '.join(data.get('languages', []))}",
        ]
        
        if data.get('availability'):
            content_parts.append(f"Availability: {data['availability']}")
        
        content = '\n'.join(content_parts)
        
        return [Chunk(
            content=content,
            category='about',
            subcategory='profile',
            technologies=None,
            metadata={
                'email': data.get('email'),
                'phone': data.get('phone'),
                'portfolio': data.get('portfolio'),
                'github': data.get('github'),
                'linkedin': data.get('linkedin')
            }
        )]
    
    def chunk_skills(self, data: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk skills data - one chunk per skill.
        
        Args:
            data: List of skill dictionaries
            
        Returns:
            List of skill chunks
        """
        chunks = []
        
        for skill in data:
            content = f"{skill.get('name', 'Unknown Skill')}: {skill.get('description', 'No description')}"
            
            if skill.get('proficiency'):
                content += f"\nProficiency: {skill['proficiency']}"
            
            chunks.append(Chunk(
                content=content,
                category='skills',
                subcategory=skill.get('category', 'General'),
                technologies=[skill.get('name', 'Unknown')],
                metadata={
                    'proficiency': skill.get('proficiency'),
                    'skill_category': skill.get('category')
                }
            ))
        
        return chunks
    
    def chunk_experience(self, data: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk experience data - one chunk per job position.
        
        Args:
            data: List of experience dictionaries
            
        Returns:
            List of experience chunks
        """
        chunks = []
        
        for exp in data:
            content_parts = [
                f"{exp.get('title', 'Unknown Title')} at {exp.get('company', 'Unknown Company')}",
                f"Duration: {exp.get('duration', 'N/A')}",
                f"Location: {exp.get('location', 'N/A')}",
                f"\n{exp.get('description', 'No description')}"
            ]
            
            content = '\n'.join(content_parts)
            
            chunks.append(Chunk(
                content=content,
                category='experience',
                subcategory=exp.get('category', 'General'),
                technologies=exp.get('key_technologies', []),
                metadata={
                    'company': exp.get('company'),
                    'title': exp.get('title'),
                    'duration': exp.get('duration'),
                    'start_date': exp.get('start_date'),
                    'end_date': exp.get('end_date'),
                    'location': exp.get('location')
                }
            ))
        
        return chunks
    
    def chunk_projects(self, data: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk projects data - one chunk per project.
        
        Args:
            data: List of project dictionaries
            
        Returns:
            List of project chunks
        """
        chunks = []
        
        for project in data:
            content_parts = [
                f"Project: {project.get('title', 'Unknown Project')}",
                f"Category: {project.get('category', 'N/A')}",
                f"Description: {project.get('description', 'No description')}",
            ]
            
            if project.get('technologies'):
                content_parts.append(f"Technologies: {', '.join(project['technologies'])}")
            
            if project.get('url'):
                content_parts.append(f"URL: {project['url']}")
            
            content = '\n'.join(content_parts)
            
            chunks.append(Chunk(
                content=content,
                category='projects',
                subcategory=project.get('category', 'General'),
                technologies=project.get('technologies', []),
                metadata={
                    'title': project.get('title'),
                    'url': project.get('url'),
                    'project_category': project.get('category')
                }
            ))
        
        return chunks
    
    def chunk_education(self, data: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk education data - one chunk per degree.
        
        Args:
            data: List of education dictionaries
            
        Returns:
            List of education chunks
        """
        chunks = []
        
        for edu in data:
            content_parts = [
                f"Degree: {edu.get('degree', 'Unknown Degree')}",
                f"Institution: {edu.get('institution', 'Unknown Institution')}",
                f"Duration: {edu.get('duration', 'N/A')}",
            ]
            
            if edu.get('score'):
                content_parts.append(f"Score: {edu['score']}")
            
            if edu.get('location'):
                content_parts.append(f"Location: {edu['location']}")
            
            content = '\n'.join(content_parts)
            
            chunks.append(Chunk(
                content=content,
                category='education',
                subcategory=edu.get('level', 'General'),
                technologies=None,
                metadata={
                    'degree': edu.get('degree'),
                    'institution': edu.get('institution'),
                    'duration': edu.get('duration'),
                    'start_date': edu.get('start_date'),
                    'end_date': edu.get('end_date'),
                    'score': edu.get('score')
                }
            ))
        
        return chunks
    
    def chunk_certifications(self, data: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk certifications data - one chunk per certification.
        
        Args:
            data: List of certification dictionaries
            
        Returns:
            List of certification chunks
        """
        chunks = []
        
        for cert in data:
            content_parts = [
                f"Certification: {cert.get('name', 'Unknown Certification')}",
                f"Provider: {cert.get('provider', 'Unknown Provider')}",
            ]
            
            if cert.get('year'):
                content_parts.append(f"Year: {cert['year']}")
            
            if cert.get('category'):
                content_parts.append(f"Category: {cert['category']}")
            
            if cert.get('url'):
                content_parts.append(f"URL: {cert['url']}")
            
            content = '\n'.join(content_parts)
            
            chunks.append(Chunk(
                content=content,
                category='certifications',
                subcategory=cert.get('category', 'General'),
                technologies=None,
                metadata={
                    'name': cert.get('name'),
                    'provider': cert.get('provider'),
                    'year': cert.get('year'),
                    'url': cert.get('url')
                }
            ))
        
        return chunks
    
    def chunk_contact(self, data: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk contact data into a single chunk.
        
        Args:
            data: Contact data dictionary
            
        Returns:
            List containing single chunk
        """
        content_parts = [
            f"Email: {data.get('email', 'N/A')}",
            f"Phone: {data.get('phone', 'N/A')}",
            f"Location: {data.get('location', 'N/A')}",
        ]
        
        if data.get('github'):
            content_parts.append(f"GitHub: {data['github']}")
        
        if data.get('linkedin'):
            content_parts.append(f"LinkedIn: {data['linkedin']}")
        
        if data.get('portfolio'):
            content_parts.append(f"Portfolio: {data['portfolio']}")
        
        if data.get('website'):
            content_parts.append(f"Website: {data['website']}")
        
        content = '\n'.join(content_parts)
        
        return [Chunk(
            content=content,
            category='contact',
            subcategory='contact_info',
            technologies=None,
            metadata=data
        )]
    
    def chunk_data(self, data: Dict[str, Any]) -> List[Chunk]:
        """
        Apply category-specific chunking strategies to loaded data.
        
        Args:
            data: Dictionary mapping categories to data
            
        Returns:
            List of all chunks across all categories
        """
        all_chunks = []
        
        for category, category_data in data.items():
            self.logger.info(f"Chunking {category} data...")
            
            try:
                if category == 'about':
                    chunks = self.chunk_about(category_data)
                elif category == 'skills':
                    chunks = self.chunk_skills(category_data)
                elif category == 'experience':
                    chunks = self.chunk_experience(category_data)
                elif category == 'projects':
                    chunks = self.chunk_projects(category_data)
                elif category == 'education':
                    chunks = self.chunk_education(category_data)
                elif category == 'certifications':
                    chunks = self.chunk_certifications(category_data)
                elif category == 'contact':
                    chunks = self.chunk_contact(category_data)
                else:
                    self.logger.warning(f"Unknown category: {category}")
                    continue
                
                all_chunks.extend(chunks)
                self.logger.info(f"Created {len(chunks)} chunks for {category}")
                
            except Exception as e:
                self.logger.error(f"Failed to chunk {category} data: {e}")
                continue
        
        return all_chunks
    
    def create_chunk(
        self,
        content: str,
        category: str,
        subcategory: Optional[str] = None,
        technologies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """
        Create a chunk object with all fields.
        
        Args:
            content: Text content of the chunk
            category: Category (e.g., "skills", "experience")
            subcategory: Optional subcategory
            technologies: Optional list of technologies
            metadata: Optional metadata dictionary
            
        Returns:
            Chunk object
        """
        return Chunk(
            content=content,
            category=category,
            subcategory=subcategory,
            technologies=technologies,
            metadata=metadata
        )
    
    def ingest(
        self,
        dry_run: bool = False,
        category_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main ingestion function.
        
        Loads data, chunks it, generates embeddings, and inserts into vector store.
        
        Args:
            dry_run: If True, preview chunks without inserting
            category_filter: Optional category to ingest (e.g., "skills")
            
        Returns:
            Dictionary with ingestion statistics
        """
        start_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Profile Data Ingestion")
        self.logger.info("=" * 60)
        
        if dry_run:
            self.logger.info("DRY RUN MODE - No data will be inserted")
        
        if category_filter:
            self.logger.info(f"Category filter: {category_filter}")
        
        # Step 1: Load JSON files
        self.logger.info("\n[1/4] Loading JSON files...")
        try:
            data = self.load_json_files(category_filter)
            self.logger.info(f"Loaded {len(data)} categories")
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return {'success': False, 'error': str(e)}
        
        # Step 2: Chunk data
        self.logger.info("\n[2/4] Chunking data...")
        try:
            chunks = self.chunk_data(data)
            self.logger.info(f"Created {len(chunks)} total chunks")
        except Exception as e:
            self.logger.error(f"Failed to chunk data: {e}")
            return {'success': False, 'error': str(e)}
        
        if dry_run:
            self.logger.info("\n=== DRY RUN PREVIEW ===")
            for i, chunk in enumerate(chunks[:5], 1):  # Show first 5
                self.logger.info(f"\nChunk {i}:")
                self.logger.info(f"  Category: {chunk.category}")
                self.logger.info(f"  Subcategory: {chunk.subcategory}")
                self.logger.info(f"  Technologies: {chunk.technologies}")
                self.logger.info(f"  Content: {chunk.content[:100]}...")
            
            if len(chunks) > 5:
                self.logger.info(f"\n... and {len(chunks) - 5} more chunks")
            
            return {
                'success': True,
                'dry_run': True,
                'total_chunks': len(chunks),
                'categories': list(data.keys())
            }
        
        # Step 3: Generate embeddings
        self.logger.info("\n[3/4] Generating embeddings...")
        try:
            # Extract content for batch embedding
            contents = [chunk.content for chunk in chunks]
            
            # Generate embeddings in batches
            self.logger.info(f"Embedding {len(contents)} chunks...")
            embeddings = self.embedding_service.embed_batch(contents, batch_size=10)
            self.logger.info(f"Generated {len(embeddings)} embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            return {'success': False, 'error': str(e)}
        
        # Step 4: Insert into vector store
        self.logger.info("\n[4/4] Inserting into vector store...")
        
        if not self.vector_store:
            self.logger.error("Vector store not initialized")
            return {'success': False, 'error': 'Vector store not available'}
        
        try:
            # Delete existing embeddings for categories being ingested
            categories_to_ingest = set(chunk.category for chunk in chunks)
            for category in categories_to_ingest:
                deleted = self.vector_store.delete_by_category(category)
                self.logger.info(f"Deleted {deleted} existing embeddings for category '{category}'")
            
            # Prepare embedding data for batch insert
            embedding_data_list = []
            for chunk, embedding in zip(chunks, embeddings):
                embedding_data = EmbeddingData(
                    content=chunk.content,
                    embedding=embedding,
                    category=chunk.category,
                    subcategory=chunk.subcategory,
                    technologies=chunk.technologies,
                    metadata=chunk.metadata
                )
                embedding_data_list.append(embedding_data)
            
            # Batch insert
            rows_inserted = self.vector_store.insert_batch(embedding_data_list)
            self.logger.info(f"Inserted {rows_inserted} embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to insert embeddings: {e}")
            return {'success': False, 'error': str(e)}
        
        # Calculate statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        stats = {
            'success': True,
            'total_chunks': len(chunks),
            'categories': list(categories_to_ingest),
            'rows_inserted': rows_inserted,
            'duration_seconds': duration
        }
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Ingestion Complete!")
        self.logger.info("=" * 60)
        self.logger.info(f"Total chunks: {stats['total_chunks']}")
        self.logger.info(f"Categories: {', '.join(stats['categories'])}")
        self.logger.info(f"Rows inserted: {stats['rows_inserted']}")
        self.logger.info(f"Duration: {stats['duration_seconds']:.2f} seconds")
        self.logger.info("=" * 60)
        
        return stats


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(
        description='Ingest profile data into vector database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all data
  python ingest_profile_data.py
  
  # Preview chunks without inserting
  python ingest_profile_data.py --dry-run
  
  # Ingest only skills data
  python ingest_profile_data.py --category skills
  
  # Use custom data directory
  python ingest_profile_data.py --data-dir /path/to/data
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview chunks without inserting into database'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        choices=['about', 'skills', 'experience', 'projects', 'education', 'certifications', 'contact'],
        help='Ingest only specific category'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Custom data directory path (default: backend/app/data/)'
    )
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Default to backend/app/data/
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / 'data'
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    
    if not args.dry_run and not database_url:
        logger.error("DATABASE_URL environment variable not set")
        logger.error("Set it with: export DATABASE_URL='postgresql://user:pass@host:5432/dbname'")
        sys.exit(1)
    
    try:
        # Initialize services
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService()
        
        vector_store = None
        if not args.dry_run:
            logger.info("Initializing vector store...")
            vector_store = VectorStoreManager(database_url)
        
        # Create ingestion handler
        ingestion = ProfileDataIngestion(
            data_dir=str(data_dir),
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        # Run ingestion
        result = ingestion.ingest(
            dry_run=args.dry_run,
            category_filter=args.category
        )
        
        if result['success']:
            sys.exit(0)
        else:
            logger.error(f"Ingestion failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
