import FacilityRiskForm from './components/FacilityRiskForm'
import StandardsSearch from './components/StandardsSearch'
import FacilityOverview from './components/FacilityOverview'
import ChatPanel from './components/ChatPanel'
import './App.css'

function App() {
  return (
    <div className="app-shell">
      <header>
        <h1>Project Intelligence Platform</h1>
        <p className="subtitle">Synthetic facility risk assessment demo — all data fabricated</p>
      </header>
      <main>
        <div className="views-column">
          <FacilityRiskForm />
          <StandardsSearch />
          <FacilityOverview />
          <ChatPanel />
        </div>
      </main>
    </div>
  )
}

export default App
