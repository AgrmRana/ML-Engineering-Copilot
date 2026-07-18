import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, Conversation, Message, Project } from '@/lib/api'
import { Send, Plus, MessageSquare } from 'lucide-react'
import { formatDateTime } from '@/lib/utils'

export default function Assistant() {
  const [selectedConversation, setSelectedConversation] = useState<number | null>(null)
  const [selectedProject, setSelectedProject] = useState<number | null>(null)
  const [message, setMessage] = useState('')
  const [showNewConversation, setShowNewConversation] = useState(false)
  const queryClient = useQueryClient()

  const { data: projects } = useQuery({
    queryKey: ['projects'],
    queryFn: () => apiClient.listProjects().then((res) => res.data),
  })

  const { data: conversations } = useQuery({
    queryKey: ['conversations', selectedProject],
    queryFn: () =>
      apiClient
        .listConversations(selectedProject || undefined)
        .then((res) => res.data),
  })

  const { data: currentConversation } = useQuery({
    queryKey: ['conversation', selectedConversation],
    queryFn: () =>
      apiClient.getConversation(selectedConversation!).then((res) => res.data),
    enabled: !!selectedConversation,
  })

  const sendMessageMutation = useMutation({
    mutationFn: (content: string) =>
      apiClient
        .sendMessage(selectedConversation!, content, false)
        .then((res) => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conversation', selectedConversation] })
      setMessage('')
    },
  })

  const createConversationMutation = useMutation({
    mutationFn: (data: { projectId?: number; title: string }) =>
      apiClient.createConversation(data.projectId, data.title).then((res) => res.data),
    onSuccess: (data) => {
      setSelectedConversation(data.id)
      setShowNewConversation(false)
      queryClient.invalidateQueries({ queryKey: ['conversations'] })
    },
  })

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault()
    if (message.trim() && selectedConversation) {
      sendMessageMutation.mutate(message)
    }
  }

  const handleNewConversation = () => {
    createConversationMutation.mutate({
      projectId: selectedProject || undefined,
      title: 'New Conversation',
    })
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">AI Assistant</h1>
        <p className="mt-2 text-gray-600">Chat with your documents</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar */}
        <div className="space-y-4">
          <div className="card">
            <h3 className="font-semibold mb-3">Filter by Project</h3>
            <select
              value={selectedProject || ''}
              onChange={(e) => setSelectedProject(e.target.value ? Number(e.target.value) : null)}
              className="input"
            >
              <option value="">All Projects</option>
              {projects?.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          </div>

          <div className="card">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold">Conversations</h3>
              <button
                onClick={handleNewConversation}
                className="text-primary-600 hover:text-primary-700"
              >
                <Plus className="h-4 w-4" />
              </button>
            </div>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {conversations && conversations.length > 0 ? (
                conversations.map((conv) => (
                  <button
                    key={conv.id}
                    onClick={() => setSelectedConversation(conv.id)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedConversation === conv.id
                        ? 'bg-primary-50 border border-primary-200'
                        : 'hover:bg-gray-50'
                    }`}
                  >
                    <div className="font-medium text-sm text-gray-900 truncate">
                      {conv.title}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {formatDateTime(conv.created_at)}
                    </div>
                  </button>
                ))
              ) : (
                <p className="text-sm text-gray-500 text-center py-4">
                  No conversations yet
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <div className="lg:col-span-3 card flex flex-col h-[600px]">
          {selectedConversation ? (
            <>
              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {currentConversation?.messages ? (
                  currentConversation.messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${
                        msg.role === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      <div
                        className={`max-w-[70%] rounded-lg p-3 ${
                          msg.role === 'user'
                            ? 'bg-primary-600 text-white'
                            : 'bg-gray-100 text-gray-900'
                        }`}
                      >
                        <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="mt-2 pt-2 border-t border-gray-300">
                            <p className="text-xs font-medium mb-1">Sources:</p>
                            <div className="space-y-1">
                              {msg.sources.map((source, idx) => (
                                <div key={idx} className="text-xs">
                                  <span className="font-medium">{source.filename}</span>
                                  <span className="ml-2 text-gray-500">
                                    (score: {source.score.toFixed(2)})
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    Loading conversation...
                  </div>
                )}
              </div>

              {/* Input */}
              <form onSubmit={handleSendMessage} className="p-4 border-t border-gray-200">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Ask a question about your documents..."
                    className="input flex-1"
                    disabled={sendMessageMutation.isPending}
                  />
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={sendMessageMutation.isPending || !message.trim()}
                  >
                    <Send className="h-4 w-4" />
                  </button>
                </div>
              </form>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <MessageSquare className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500 mb-4">Select or create a conversation</p>
                <button
                  onClick={handleNewConversation}
                  className="btn btn-primary"
                >
                  New Conversation
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
