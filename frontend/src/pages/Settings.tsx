import React, { useState } from 'react'

export default function Settings() {
  type Tab = 'general' | 'api' | 'advanced'
  const [activeTab, setActiveTab] = useState<Tab>('general')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
        <p className="mt-2 text-gray-600">Configure application settings</p>
      </div>

      <div className="card">
        <div className="border-b border-gray-200">
          <nav className="flex gap-4">
            {(['general', 'api', 'advanced'] as Tab[]).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-3 border-b-2 font-medium transition-colors ${
                  activeTab === tab
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'general' && (
            <div className="space-y-4">
              <h3 className="font-semibold">General Settings</h3>
              <p className="text-gray-500">General application settings will be available here.</p>
            </div>
          )}
          {activeTab === 'api' && (
            <div className="space-y-4">
              <h3 className="font-semibold">API Configuration</h3>
              <p className="text-gray-500">API settings are configured on the backend server.</p>
            </div>
          )}
          {activeTab === 'advanced' && (
            <div className="space-y-4">
              <h3 className="font-semibold">Advanced Settings</h3>
              <p className="text-gray-500">Advanced configuration options coming soon.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
