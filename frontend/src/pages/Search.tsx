import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiClient, Project, SearchResult } from '@/lib/api'
import { Search as SearchIcon, FileText } from 'lucide-react'

export default function Search() {
  const [query, setQuery] = useState('')
  const [selectedProject, setSelectedProject] = useState<number | null>(null)
  const [hasSearched, setHasSearched] = useState(false)

  const { data: projects } = useQuery({
    queryKey: ['projects'],
    queryFn: () => apiClient.listProjects().then((res) => res.data),
  })

  const { data: results, isLoading } = useQuery({
    queryKey: ['search', query, selectedProject],
    queryFn: () =>
      apiClient
        .search(query, selectedProject || undefined)
        .then((res) => res.data),
    enabled: hasSearched && query.length > 0,
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      setHasSearched(true)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Search</h1>
        <p className="mt-2 text-gray-600">Search across all your documents</p>
      </div>

      {/* Search Form */}
      <div className="card">
        <form onSubmit={handleSearch} className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search for anything..."
                className="input text-lg"
              />
            </div>
            <button type="submit" className="btn btn-primary flex items-center gap-2">
              <SearchIcon className="h-4 w-4" />
              Search
            </button>
          </div>

          {projects && projects.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Filter by Project (optional)
              </label>
              <select
                value={selectedProject || ''}
                onChange={(e) => setSelectedProject(e.target.value ? Number(e.target.value) : null)}
                className="input"
              >
                <option value="">All Projects</option>
                {projects.map((project) => (
                  <option key={project.id} value={project.id}>
                    {project.name}
                  </option>
                ))}
              </select>
            </div>
          )}
        </form>
      </div>

      {/* Results */}
      {hasSearched && (
        <div className="space-y-4">
          {isLoading ? (
            <div className="text-center py-8 text-gray-500">Searching...</div>
          ) : results && results.length > 0 ? (
            <>
              <p className="text-sm text-gray-600">
                Found {results.length} result(s)
              </p>
              <div className="space-y-4">
                {results.map((result, index) => (
                  <div key={index} className="card">
                    <div className="flex items-start gap-4">
                      <div className="p-2 bg-primary-100 rounded-lg">
                        <FileText className="h-5 w-5 text-primary-600" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="font-medium text-gray-900">
                            {result.metadata.filename || 'Unknown File'}
                          </h3>
                          <span className="text-sm text-gray-500">
                            Score: {result.score.toFixed(3)}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 line-clamp-3">
                          {result.content}
                        </p>
                        <div className="flex gap-2 mt-2">
                          {result.metadata.file_type && (
                            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                              {result.metadata.file_type.toUpperCase()}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="card text-center py-12">
              <SearchIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No results found</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
