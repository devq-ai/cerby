# Cerby Identity Automation Platform - Proof of Concept

A demonstration platform showcasing automated identity management across disconnected SaaS applications with genetic algorithm policy optimization.

## 🚀 Overview

This proof of concept demonstrates:
- **Multi-SaaS Identity Management**: Ingestion from 10+ disconnected identity providers
- **Genetic Algorithm Policy Optimization**: Using Darwin to evolve optimal access policies
- **Real-time Analytics**: Stream processing with < 2 second dashboard latency
- **Compliance Automation**: SOX and GDPR compliance checking and reporting
- **Scalable Architecture**: Handles 10K+ identity events per minute

## 🛠️ Technology Stack

### Core Framework
- **FastAPI**: Modern async web framework for REST APIs
- **Logfire**: Comprehensive observability and monitoring
- **PyTest**: Test-driven development with 90%+ coverage
- **TaskMaster AI**: Task-driven development workflow
- **MCP Integration**: AI-enhanced development tools

### Data & Analytics
- **Pandas**: Data manipulation and analysis
- **Darwin/PyGAD**: Genetic algorithm optimization
- **Panel**: Interactive real-time dashboards
- **SQLAlchemy**: Database ORM with async support
- **Redis**: Caching and real-time pub/sub

### Identity Simulation
- **10+ Provider Support**: Okta, Azure AD, Google Workspace, Slack, GitHub, Jira, Confluence, Salesforce, Box, Dropbox
- **SCIM 2.0**: Standards-compliant identity provisioning
- **Webhook Support**: Real-time event processing
- **Batch Import**: CSV/JSON bulk identity loading

## 📋 Prerequisites

- Python 3.12+
- Redis (for caching and pub/sub)
- Node.js 18+ (for TaskMaster AI)
- Git

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/devq-ai/cerby-identity-automation.git
   cd cerby-identity-automation
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize the database**
   ```bash
   alembic upgrade head
   ```

6. **Start Redis** (if not already running)
   ```bash
   redis-server
   ```

## 🏃 Running the Application

### Development Mode
```bash
# Start the FastAPI server
python main.py

# In another terminal, start the Panel dashboard
panel serve src/dashboard/app.py --port 5006
```

### Production Mode
```bash
# Using Uvicorn with multiple workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Or using Docker
docker-compose up
```

## 📊 Dashboard Access

- **API Documentation**: http://localhost:8000/api/docs
- **Interactive Dashboard**: http://localhost:5006
- **Health Check**: http://localhost:8000/health

## 🧬 Genetic Algorithm Configuration

The Darwin genetic algorithm can be configured via environment variables:

```env
DARWIN_POPULATION_SIZE=100
DARWIN_GENERATIONS=50
DARWIN_MUTATION_RATE=0.1
DARWIN_CROSSOVER_RATE=0.8
DARWIN_ELITE_SIZE=10
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow         # Performance tests

# Run tests in watch mode
pytest-watch
```

## 📁 Project Structure

```
cerby/
├── src/
│   ├── api/              # REST API endpoints
│   ├── core/             # Core configuration and security
│   ├── db/               # Database models and sessions
│   ├── genetic_algorithm/# Darwin GA implementation
│   ├── ingestion/        # Identity data ingestion
│   ├── analytics/        # Real-time analytics
│   └── dashboard/        # Panel dashboard
├── tests/                # Comprehensive test suite
├── docs/                 # Additional documentation
├── .taskmaster/          # TaskMaster AI configuration
├── main.py              # Application entry point
└── requirements.txt     # Python dependencies
```

## 🔄 Development Workflow

This project uses TaskMaster AI for task-driven development:

```bash
# View current tasks
npx task-master-ai list

# Get next priority task
npx task-master-ai next

# Expand a task into subtasks
npx task-master-ai expand --id=1 --research

# Update task status
npx task-master-ai set-status --id=1 --status=done
```

## 📈 Performance Targets

- **Event Processing**: 10,000+ events per minute
- **Dashboard Latency**: < 2 seconds for real-time updates
- **API Response Time**: < 100ms for 95th percentile
- **Policy Optimization**: Converge within 50 generations
- **Test Coverage**: 90%+ line coverage

## 🔒 Security & Compliance

- **SOX Compliance**: Automated audit trails and separation of duties
- **GDPR Compliance**: Data privacy controls and right to erasure
- **Authentication**: JWT-based with refresh tokens
- **Rate Limiting**: Configurable per-endpoint limits
- **Data Encryption**: At rest and in transit

## 🚢 Deployment

### Docker Deployment
```bash
docker build -t cerby-identity-automation .
docker run -p 8000:8000 -p 5006:5006 cerby-identity-automation
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

### Infrastructure as Code (Pulumi)
```bash
cd infrastructure/
pulumi up
```

## 📝 API Examples

### Create Identity
```bash
curl -X POST http://localhost:8000/api/v1/identity \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "okta",
    "external_id": "00u1234567890",
    "email": "user@example.com",
    "attributes": {
      "department": "Engineering"
    }
  }'
```

### Start Policy Optimization
```bash
curl -X POST http://localhost:8000/api/v1/genetic-algorithm/start \
  -H "Content-Type: application/json" \
  -d '{
    "objectives": ["security", "productivity", "compliance"],
    "population_size": 100,
    "generations": 50
  }'
```

## 🤝 Contributing

1. Follow the existing code style (Black formatter, 88 char lines)
2. Write tests for new features (maintain 90%+ coverage)
3. Update documentation as needed
4. Use TaskMaster AI for task management

## 📄 License

This is a proof of concept demonstration project.

## 👥 Team

Built by the DevQ.ai Team for Cerby's Senior Data Engineer position demonstration.

## 🔗 Links

- [API Documentation](http://localhost:8000/api/docs)
- [Dashboard](http://localhost:5006)
- [TaskMaster AI Tasks](.taskmaster/tasks.json)
- [Architecture Diagram](docs/architecture.md)