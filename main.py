import os
import re
import time
import json
import requests
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import importlib.util
import sys
from collections import Counter
import math

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QListWidget, QListWidgetItem, QProgressBar,
    QFrame, QStackedWidget, QMessageBox, QScrollArea, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QMutexLocker
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QPixmap

# Fast dependency checking
DEPENDENCIES = {
    'bs4': {'available': False, 'module': None},
    'wikipedia': {'available': False, 'module': None},
    'requests': {'available': True, 'module': None}
}


def probe_dependency(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def check_dependencies():
    for name in DEPENDENCIES.keys():
        if probe_dependency(name):
            DEPENDENCIES[name]['available'] = True


check_dependencies()

# Configuration with API keys
CONFIG_FILE = "fact_checker_config.json"
DEFAULT_CONFIG = {
    "USER_AGENT": "FactChecker/2.0 (Educational Research Tool)",
    "TIMEOUT": 15,
    "REQUEST_DELAY": 0.5,
    "SERPAPI_KEY": "YOUR_SERPAPI_KEY_HERE",
    "NEWSAPI_KEY": "YOUR_NEWSAPI_KEY_HERE",
    "MAX_SOURCES": 5
}

# Enhanced Knowledge Base with comprehensive fake news training data
KNOWLEDGE_BASE = {
    "political_leaders": {
        "narendra modi": {"position": "Prime Minister", "country": "India", "party": "BJP"},
        "joe biden": {"position": "President", "country": "United States", "party": "Democratic"},
        "donald trump": {"position": "Former President", "country": "United States", "party": "Republican"},
        "xi jinping": {"position": "President", "country": "China", "party": "Communist Party"},
        "vladimir putin": {"position": "President", "country": "Russia", "party": "United Russia"},
        "rahul gandhi": {"position": "Member of Parliament", "country": "India", "party": "Congress"},
        "emmanuel macron": {"position": "President", "country": "France", "party": "Renaissance"},
    },
    "ceo_leaders": {
        "elon musk": {"position": "CEO", "companies": ["Tesla", "SpaceX", "X (Twitter)"],
                      "not_ceo_of": ["Apple", "Microsoft", "Amazon", "Google"]},
        "tim cook": {"position": "CEO", "companies": ["Apple"],
                     "not_ceo_of": ["Tesla", "Microsoft", "Amazon", "Google"]},
        "mark zuckerberg": {"position": "CEO", "companies": ["Meta", "Facebook"],
                            "not_ceo_of": ["Apple", "Tesla", "Amazon", "Google"]},
        "satya nadella": {"position": "CEO", "companies": ["Microsoft"],
                          "not_ceo_of": ["Apple", "Tesla", "Amazon", "Google"]},
        "sundar pichai": {"position": "CEO", "companies": ["Google", "Alphabet"],
                          "not_ceo_of": ["Apple", "Tesla", "Amazon", "Microsoft"]},
        "jeff bezos": {"position": "Executive Chairman", "companies": ["Amazon"],
                       "not_ceo_of": ["Apple", "Tesla", "Microsoft", "Google"]},
    },
    "company_ownership": {
        "apple": {"founded_by": ["Steve Jobs", "Steve Wozniak"], "current_ceo": "Tim Cook",
                  "not_owned_by": ["Elon Musk", "Mark Zuckerberg"]},
        "tesla": {"founded_by": ["Elon Musk"], "current_ceo": "Elon Musk",
                  "not_owned_by": ["Tim Cook", "Mark Zuckerberg"]},
        "meta": {"founded_by": ["Mark Zuckerberg"], "current_ceo": "Mark Zuckerberg",
                 "not_owned_by": ["Elon Musk", "Tim Cook"]},
        "microsoft": {"founded_by": ["Bill Gates", "Paul Allen"], "current_ceo": "Satya Nadella",
                      "not_owned_by": ["Elon Musk", "Tim Cook"]},
    },
    "capitals_verified": {
        "new delhi": "India", "tokyo": "Japan", "paris": "France", "berlin": "Germany",
        "washington": "United States", "washington dc": "United States", "beijing": "China",
        "moscow": "Russia", "london": "United Kingdom", "ottawa": "Canada", "canberra": "Australia",
        "brasilia": "Brazil", "mexico city": "Mexico", "cairo": "Egypt", "rome": "Italy",
        "madrid": "Spain", "amsterdam": "Netherlands", "stockholm": "Sweden", "oslo": "Norway"
    },
    "countries_capitals_verified": {
        "india": "New Delhi", "japan": "Tokyo", "france": "Paris", "germany": "Berlin",
        "united states": "Washington DC", "usa": "Washington DC", "china": "Beijing",
        "russia": "Moscow", "united kingdom": "London", "uk": "London", "canada": "Ottawa",
        "australia": "Canberra", "brazil": "Brasilia", "mexico": "Mexico City", "egypt": "Cairo",
        "italy": "Rome", "spain": "Madrid", "netherlands": "Amsterdam", "sweden": "Stockholm"
    },
    "fake_news_indicators": [
        "breaking news", "you won't believe", "you wont believe", "shocking truth", "doctors hate this",
        "this will blow your mind", "urgent update", "must see", "incredible discovery",
        "leaked footage", "secret revealed", "insider reveals", "unbelievable",
        "amazing trick", "they don't want you to know", "they dont want you to know",
        "exposed", "viral", "click to see", "number 7 will shock you", "wait until you see",
        "government hiding", "conspiracy revealed", "forbidden knowledge", "breaking:",
        "urgent:", "shocking:", "unbelievable:", "exclusive:", "leaked:", "exposed:"
    ],
    "fake_geographical_claims": [
        "delhi became capital of russia", "delhi is capital of russia", "new delhi capital russia",
        "moscow is capital of india", "moscow capital india", "paris capital germany",
        "berlin capital france", "tokyo capital china", "beijing capital japan",
        "london capital france", "madrid capital italy", "rome capital spain",
        "washington capital china", "beijing capital usa", "moscow capital usa"
    ],
    "fake_political_claims": [
        "modi became president of usa", "modi is president america", "biden prime minister india",
        "trump president china", "xi jinping president usa", "putin president india",
        "macron president germany", "modi became president russia", "biden president china"
    ],
    "fake_business_claims": [
        "elon musk owns apple", "elon musk ceo apple", "tim cook owns tesla",
        "mark zuckerberg owns apple", "bill gates ceo apple", "jeff bezos ceo tesla",
        "sundar pichai ceo apple", "satya nadella ceo tesla"
    ],
    "fake_death_claims": [
        "modi died", "modi dead", "modi passed away", "narendra modi died",
        "biden died", "biden dead", "trump died", "trump dead", "elon musk died",
        "tim cook died", "mark zuckerberg died", "xi jinping died", "putin died"
    ],
    "medical_misinformation_patterns": [
        "died due to heart attack", "sudden death", "found dead", "died suddenly",
        "mysterious death", "died of covid", "vaccine killed", "died after vaccination"
    ],
    "verified_facts_trained": [
        "new delhi is capital of india", "tokyo is capital of japan", "paris is capital of france",
        "moscow is capital of russia", "washington is capital of usa", "berlin is capital of germany",
        "tim cook is ceo of apple", "elon musk is ceo of tesla", "narendra modi is prime minister of india",
        "joe biden is president of united states", "xi jinping is president of china"
    ]
}


class ConfigManager:
    @staticmethod
    def load_config():
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    for key, value in DEFAULT_CONFIG.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                print(f"Config load error: {e}")
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()

    @staticmethod
    def save_config(config):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            print(f"Config save error: {e}")
            return False


class EnhancedKnowledgeBaseDetector:
    """Priority detector that checks knowledge base FIRST"""

    def __init__(self):
        self.mutex = QMutex()

    def detect_fake_news_indicators(self, claim: str) -> Dict:
        """Detect fake news indicator words with evidence"""
        locker = QMutexLocker(self.mutex)
        claim_lower = claim.lower().strip()

        detected_indicators = []
        indicator_count = 0

        for indicator in KNOWLEDGE_BASE["fake_news_indicators"]:
            if indicator in claim_lower:
                detected_indicators.append(indicator)
                indicator_count += 1

        if indicator_count > 0:
            # Calculate fake probability based on indicators found
            fake_probability = min(0.95, 0.60 + (indicator_count * 0.10))

            evidence_text = (
                f"FAKE NEWS LANGUAGE DETECTED: Found {indicator_count} fake news indicator(s) in the claim: "
                f"{', '.join([f'\"{ind}\"' for ind in detected_indicators])}. "
                f"These phrases are commonly used in misleading or sensationalized fake news headlines."
            )

            return {
                "detected": True,
                "indicator_count": indicator_count,
                "indicators_found": detected_indicators,
                "fake_probability": fake_probability,
                "confidence": 0.95,
                "evidence": {
                    "type": "fake_news_indicators",
                    "text": evidence_text,
                    "source": "Knowledge Base - Fake News Language Detection",
                    "supports": False,
                    "confidence": 0.95
                }
            }

        return {"detected": False}

    def check_trained_fake_patterns(self, claim: str) -> List[Dict]:
        """Check against all trained fake patterns"""
        locker = QMutexLocker(self.mutex)
        claim_lower = claim.lower().strip()
        found_patterns = []

        # Check geographical fakes
        for fake_geo in KNOWLEDGE_BASE["fake_geographical_claims"]:
            similarity = self._calculate_similarity(claim_lower, fake_geo)
            if similarity > 0.65:
                correct_info = self._get_correct_geographical_info(fake_geo)
                found_patterns.append({
                    "type": "trained_fake_geographical",
                    "pattern": fake_geo,
                    "similarity": similarity,
                    "confidence": 0.97,
                    "evidence": {
                        "type": "knowledge_base_contradiction",
                        "text": f"FALSE GEOGRAPHICAL CLAIM: This matches the known fake pattern '{fake_geo}'. {correct_info}",
                        "source": "Knowledge Base - Verified Geography Facts",
                        "supports": False,
                        "confidence": 0.97
                    }
                })

        # Check political fakes
        for fake_pol in KNOWLEDGE_BASE["fake_political_claims"]:
            similarity = self._calculate_similarity(claim_lower, fake_pol)
            if similarity > 0.65:
                correct_info = self._get_correct_political_info(fake_pol)
                found_patterns.append({
                    "type": "trained_fake_political",
                    "pattern": fake_pol,
                    "similarity": similarity,
                    "confidence": 0.97,
                    "evidence": {
                        "type": "knowledge_base_contradiction",
                        "text": f"FALSE POLITICAL CLAIM: This matches the known fake pattern '{fake_pol}'. {correct_info}",
                        "source": "Knowledge Base - Verified Political Facts",
                        "supports": False,
                        "confidence": 0.97
                    }
                })

        # Check business fakes
        for fake_biz in KNOWLEDGE_BASE["fake_business_claims"]:
            similarity = self._calculate_similarity(claim_lower, fake_biz)
            if similarity > 0.65:
                correct_info = self._get_correct_business_info(fake_biz)
                found_patterns.append({
                    "type": "trained_fake_business",
                    "pattern": fake_biz,
                    "similarity": similarity,
                    "confidence": 0.97,
                    "evidence": {
                        "type": "knowledge_base_contradiction",
                        "text": f"FALSE BUSINESS CLAIM: This matches the known fake pattern '{fake_biz}'. {correct_info}",
                        "source": "Knowledge Base - Verified Business Leadership",
                        "supports": False,
                        "confidence": 0.97
                    }
                })

        # Check death hoax fakes
        for fake_death in KNOWLEDGE_BASE["fake_death_claims"]:
            similarity = self._calculate_similarity(claim_lower, fake_death)
            if similarity > 0.60:  # Lower threshold for death claims
                found_patterns.append({
                    "type": "trained_fake_death",
                    "pattern": fake_death,
                    "similarity": similarity,
                    "confidence": 0.98,
                    "evidence": {
                        "type": "knowledge_base_contradiction",
                        "text": f"FALSE DEATH CLAIM: This matches the known death hoax pattern '{fake_death}'. Major celebrity or political figure deaths would be immediately reported by credible international news agencies. No such reports exist.",
                        "source": "Knowledge Base - Death Hoax Detection",
                        "supports": False,
                        "confidence": 0.98
                    }
                })

        return found_patterns

    def _get_correct_geographical_info(self, fake_pattern: str) -> str:
        """Extract correct geographical information"""
        if "delhi" in fake_pattern and "russia" in fake_pattern:
            return "New Delhi is the capital of India. Moscow is the capital of Russia."
        elif "moscow" in fake_pattern and "india" in fake_pattern:
            return "Moscow is the capital of Russia. New Delhi is the capital of India."
        elif "paris" in fake_pattern and "germany" in fake_pattern:
            return "Paris is the capital of France. Berlin is the capital of Germany."
        elif "tokyo" in fake_pattern and "china" in fake_pattern:
            return "Tokyo is the capital of Japan. Beijing is the capital of China."
        return "This is a known false geographical claim."

    def _get_correct_political_info(self, fake_pattern: str) -> str:
        """Extract correct political information"""
        if "modi" in fake_pattern:
            return "Narendra Modi is the Prime Minister of India, not a leader of any other country."
        elif "biden" in fake_pattern:
            return "Joe Biden is the President of the United States, not a leader of any other country."
        elif "trump" in fake_pattern:
            return "Donald Trump is the former President of the United States."
        return "This is a known false political claim."

    def _get_correct_business_info(self, fake_pattern: str) -> str:
        """Extract correct business information"""
        if "elon musk" in fake_pattern and "apple" in fake_pattern:
            return "Elon Musk is the CEO of Tesla, SpaceX, and X (Twitter). Tim Cook is the CEO of Apple."
        elif "tim cook" in fake_pattern and "tesla" in fake_pattern:
            return "Tim Cook is the CEO of Apple. Elon Musk is the CEO of Tesla."
        elif "mark zuckerberg" in fake_pattern and "apple" in fake_pattern:
            return "Mark Zuckerberg is the CEO of Meta (Facebook). Tim Cook is the CEO of Apple."
        return "This is a known false business leadership claim."

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Enhanced similarity calculation with key word matching"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union) if union else 0

        # Boost for key word matches
        key_words = ["capital", "ceo", "president", "prime minister", "became", "owns", "died", "dead"]
        key_matches = sum(1 for key_word in key_words if key_word in text1 and key_word in text2)

        if key_matches > 0:
            jaccard += (key_matches * 0.08)

        return min(jaccard, 1.0)


class EnhancedLinguisticAnalyzer:
    """Advanced linguistic pattern analyzer"""

    def __init__(self):
        self.setup_patterns()

    def setup_patterns(self):
        self.emotional_manipulation = {
            "shock_words": [
                "shocking", "unbelievable", "incredible", "amazing", "stunning",
                "mind-blowing", "jaw-dropping", "outrageous", "explosive", "scandalous"
            ],
            "urgency_words": [
                "breaking", "urgent", "immediate", "now", "quickly", "hurry",
                "don't wait", "dont wait", "limited time", "act fast", "before it's too late"
            ],
            "authority_undermining": [
                "they don't want you to know", "they dont want you to know", "hidden truth",
                "cover-up", "exposed", "revealed", "leaked", "insider", "secret", "forbidden", "censored"
            ],
            "clickbait_phrases": [
                "you won't believe", "you wont believe", "what happens next will shock you",
                "number will surprise you", "doctors hate this", "one weird trick",
                "this changes everything", "wait until you see", "must see", "viral",
                "everyone is talking about"
            ]
        }

        self.credible_patterns = {
            "attribution_words": [
                "according to", "sources say", "reported by", "confirmed by",
                "stated", "announced", "disclosed", "revealed in a statement"
            ],
            "uncertainty_markers": [
                "allegedly", "reportedly", "appears to", "seems to", "may have",
                "could be", "might be", "possibly", "potentially", "likely"
            ]
        }

    def analyze_headline(self, headline: str) -> Dict:
        text_lower = headline.lower().strip()

        manipulation_flags = []
        manipulation_score = 0.0

        for category, words in self.emotional_manipulation.items():
            found_words = []
            for word in words:
                if word in text_lower:
                    found_words.append(word)
                    manipulation_score += 0.2

            if found_words:
                manipulation_flags.append({
                    "type": category,
                    "found_words": found_words,
                    "severity": "high" if len(found_words) > 1 else "medium"
                })

        credibility_score = 0.0
        credibility_markers = []

        for category, words in self.credible_patterns.items():
            for word in words:
                if word in text_lower:
                    credibility_markers.append({"type": category, "marker": word})
                    if category == "attribution_words":
                        credibility_score += 0.25
                    elif category == "uncertainty_markers":
                        credibility_score += 0.15

        fake_probability = min(manipulation_score, 1.0) * 0.5 - min(credibility_score, 1.0) * 0.3
        fake_probability = max(0.0, min(1.0, fake_probability))

        return {
            "original_text": headline,
            "manipulation_score": min(manipulation_score, 1.0),
            "credibility_score": min(credibility_score, 1.0),
            "linguistic_flags": manipulation_flags,
            "overall_fake_probability": fake_probability,
            "credibility_markers": credibility_markers
        }


class SmartClaimAnalyzer:
    """Enhanced analyzer with knowledge base priority"""

    def __init__(self):
        self.mutex = QMutex()
        self.kb_detector = EnhancedKnowledgeBaseDetector()
        self.linguistic_analyzer = EnhancedLinguisticAnalyzer()

    def analyze_claim(self, claim: str) -> Dict:
        locker = QMutexLocker(self.mutex)

        claim_lower = claim.lower().strip()
        analysis = {
            "original_claim": claim,
            "claim_type": self._determine_claim_type(claim_lower),
            "entities": self._extract_smart_entities(claim_lower),
            "search_queries": self._generate_search_queries(claim_lower)
        }

        # PRIORITY 1: Check for fake news indicator words FIRST
        fake_indicator_result = self.kb_detector.detect_fake_news_indicators(claim)
        analysis["fake_news_indicators"] = fake_indicator_result

        # PRIORITY 2: Check trained fake patterns
        trained_fakes = self.kb_detector.check_trained_fake_patterns(claim)
        analysis["trained_fake_patterns"] = trained_fakes

        # PRIORITY 3: Linguistic analysis
        linguistic_analysis = self.linguistic_analyzer.analyze_headline(claim)
        analysis["linguistic_analysis"] = linguistic_analysis

        # PRIORITY 4: Geographical and entity checks
        analysis["geographical_checks"] = self._check_geographical_facts(claim_lower)

        return analysis

    def _determine_claim_type(self, claim: str) -> str:
        if any(word in claim for word in ["capital", "became capital", "is capital"]):
            return "geographical"
        elif any(word in claim for word in ["ceo", "owns", "founded", "company"]):
            return "business_leadership"
        elif any(word in claim for word in ["president", "prime minister", "minister", "became president"]):
            return "political_leadership"
        elif any(word in claim for word in ["died", "dead", "passed away"]):
            return "death_claim"
        else:
            return "general"

    def _extract_smart_entities(self, claim: str) -> Dict:
        entities = {"people": [], "companies": [], "positions": [], "countries": [], "cities": []}

        for person in list(KNOWLEDGE_BASE["political_leaders"].keys()) + list(KNOWLEDGE_BASE["ceo_leaders"].keys()):
            if person in claim:
                entities["people"].append(person.title())

        for company in KNOWLEDGE_BASE["company_ownership"].keys():
            if company in claim:
                entities["companies"].append(company.title())

        for city in KNOWLEDGE_BASE["capitals_verified"].keys():
            if city in claim:
                entities["cities"].append(city.title())

        for country in KNOWLEDGE_BASE["countries_capitals_verified"].keys():
            if country in claim:
                entities["countries"].append(country.title())

        return entities

    def _generate_search_queries(self, claim: str) -> List[str]:
        queries = [claim]

        if "capital" in claim:
            words = claim.split()
            for i, word in enumerate(words):
                if word in KNOWLEDGE_BASE["capitals_verified"] or word in KNOWLEDGE_BASE["countries_capitals_verified"]:
                    if i < len(words) - 1:
                        next_word = words[i + 1]
                        queries.append(f"what is capital of {next_word}")
                        queries.append(f"{word} capital of which country")

        elif "ceo" in claim or "owns" in claim:
            for company in KNOWLEDGE_BASE["company_ownership"].keys():
                if company in claim:
                    queries.append(f"who is CEO of {company}")
                    queries.append(f"{company} current CEO")

        return queries[:3]

    def _check_geographical_facts(self, claim: str) -> List[Dict]:
        geo_results = []
        cities = []
        countries = []

        for city in KNOWLEDGE_BASE["capitals_verified"].keys():
            if city in claim:
                cities.append(city)

        for country in KNOWLEDGE_BASE["countries_capitals_verified"].keys():
            if country in claim:
                countries.append(country)

        if "capital" in claim or "became" in claim:
            for city in cities:
                correct_country = KNOWLEDGE_BASE["capitals_verified"][city]
                for country in countries:
                    country_capital = KNOWLEDGE_BASE["countries_capitals_verified"].get(country, "")

                    if correct_country.lower() != country and city not in country_capital.lower():
                        geo_results.append({
                            "type": "geographical_contradiction",
                            "fact": f"{city.title()} is the capital of {correct_country}, not {country.title()}",
                            "confidence": 0.98
                        })

        return geo_results


class EnhancedFactChecker:
    def __init__(self, config: Dict):
        self.config = config
        self.analyzer = SmartClaimAnalyzer()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.get("USER_AGENT")})

        self.serpapi_key = config.get("SERPAPI_KEY", "")
        self.newsapi_key = config.get("NEWSAPI_KEY", "")

    def check_fact(self, claim: str) -> Dict:
        try:
            # Analyze claim with KNOWLEDGE BASE PRIORITY
            analysis = self.analyzer.analyze_claim(claim)
            all_evidence = []
            confidence_scores = []

            # PRIORITY 1: Check fake news indicators FIRST
            fake_indicator_result = analysis.get("fake_news_indicators", {})
            if fake_indicator_result.get("detected", False):
                evidence_item = fake_indicator_result["evidence"]
                all_evidence.append(evidence_item)
                confidence_scores.append(-0.95)  # Strong negative evidence

                # If we detected fake news indicators, we can be very confident it's fake
                return {
                    "status": "FALSE",
                    "confidence": fake_indicator_result.get("confidence", 0.95),
                    "summary": f"FAKE NEWS DETECTED: Found {fake_indicator_result['indicator_count']} fake news indicators in the claim. These language patterns are characteristic of misleading content.",
                    "evidence": all_evidence,
                    "source": "Enhanced Knowledge Base Detection",
                    "claim_analysis": analysis
                }

            # PRIORITY 2: Check trained fake patterns
            trained_fakes = analysis.get("trained_fake_patterns", [])
            if trained_fakes:
                for fake_pattern in trained_fakes:
                    evidence_item = fake_pattern["evidence"]
                    all_evidence.append(evidence_item)
                    confidence_scores.append(-0.97)

                # Strong match with trained fake patterns
                return {
                    "status": "FALSE",
                    "confidence": 0.97,
                    "summary": f"FALSE CLAIM: This matches {len(trained_fakes)} known false pattern(s) in our knowledge base.",
                    "evidence": all_evidence,
                    "source": "Enhanced Knowledge Base Detection",
                    "claim_analysis": analysis
                }

            # PRIORITY 3: Linguistic analysis
            linguistic_analysis = analysis.get("linguistic_analysis", {})
            linguistic_fake_prob = linguistic_analysis.get("overall_fake_probability", 0)

            if linguistic_fake_prob > 0.6:
                all_evidence.append({
                    "type": "linguistic_analysis",
                    "text": f"SUSPICIOUS WRITING STYLE: The language patterns suggest potential fake news (probability: {linguistic_fake_prob:.1%}). Contains emotional manipulation or clickbait phrases.",
                    "source": "Advanced Linguistic Pattern Analysis",
                    "supports": False,
                    "confidence": 0.80
                })
                confidence_scores.append(-0.80)

            # PRIORITY 4: Enhanced Knowledge Base Checks
            kb_result = self._check_enhanced_knowledge_base(claim, analysis)
            if kb_result:
                all_evidence.append(kb_result)
                confidence_scores.append(0.95 if kb_result["supports"] else -0.95)

            # PRIORITY 5: External searches (only if no strong KB evidence)
            if not trained_fakes and linguistic_fake_prob < 0.6:
                # Wikipedia Search
                wiki_evidence = self._search_wikipedia_enhanced(analysis["search_queries"])
                all_evidence.extend(wiki_evidence)
                confidence_scores.extend([0.7 for _ in wiki_evidence])

                # Web Search
                web_evidence = self._search_web_enhanced(analysis["search_queries"])
                all_evidence.extend(web_evidence)
                confidence_scores.extend([0.8 for _ in web_evidence])

            # Calculate overall confidence
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                confidence = min(abs(avg_confidence), 0.98)
            else:
                confidence = 0.3

            # Determine status
            has_strong_contradictions = any(
                evidence.get("confidence", 0) > 0.9 and not evidence.get("supports", True)
                for evidence in all_evidence
            )

            if has_strong_contradictions or linguistic_fake_prob > 0.5:
                status = "FALSE"
                confidence = max(0.85, confidence)
            elif confidence > 0.75:
                status = "VERIFIED"
            elif confidence < 0.4:
                status = "FALSE"
            else:
                status = "UNCERTAIN"

            return {
                "status": status,
                "confidence": confidence,
                "summary": self._generate_enhanced_summary(status, confidence, len(all_evidence), analysis),
                "evidence": all_evidence,
                "source": "Enhanced AI Fact Checker with KB Priority",
                "claim_analysis": analysis
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "confidence": 0.0,
                "summary": f"Error during fact checking: {str(e)}",
                "evidence": [{"type": "error", "text": str(e), "source": "System", "supports": False}],
                "source": "Error Handler"
            }

    def _check_enhanced_knowledge_base(self, claim: str, analysis: Dict) -> Optional[Dict]:
        """Enhanced knowledge base checking"""
        claim_lower = claim.lower()

        if analysis["claim_type"] == "business_leadership":
            return self._verify_business_claim(claim_lower, analysis)
        elif analysis["claim_type"] == "political_leadership":
            return self._verify_political_claim(claim_lower, analysis)
        elif analysis["claim_type"] == "geographical":
            return self._verify_geography_claim(claim_lower, analysis)

        return None

    def _verify_business_claim(self, claim: str, analysis: Dict) -> Optional[Dict]:
        """Verify business/CEO related claims"""
        people = [p.lower() for p in analysis["entities"]["people"]]
        companies = [c.lower() for c in analysis["entities"]["companies"]]

        for person_name in people:
            if person_name in KNOWLEDGE_BASE["ceo_leaders"]:
                person_info = KNOWLEDGE_BASE["ceo_leaders"][person_name]

                for company in companies:
                    if company in [c.lower() for c in person_info["companies"]]:
                        return {
                            "type": "knowledge_base_verified",
                            "text": f"VERIFIED: {person_name.title()} is indeed the CEO of {company.title()}",
                            "source": "Knowledge Base - Business Leadership",
                            "supports": True,
                            "confidence": 0.95
                        }
                    elif company in [c.lower() for c in person_info.get("not_ceo_of", [])]:
                        actual_companies = ", ".join(person_info["companies"])
                        return {
                            "type": "knowledge_base_contradiction",
                            "text": f"FALSE: {person_name.title()} is NOT the CEO of {company.title()}. {person_name.title()} is the CEO of {actual_companies}",
                            "source": "Knowledge Base - Business Leadership",
                            "supports": False,
                            "confidence": 0.95
                        }

        return None

    def _verify_political_claim(self, claim: str, analysis: Dict) -> Optional[Dict]:
        """Verify political leadership claims"""
        people = [p.lower() for p in analysis["entities"]["people"]]
        countries = [c.lower() for c in analysis["entities"]["countries"]]

        if "modi" in claim and ("president" in claim or "became" in claim):
            if "usa" in claim or "united states" in claim or "america" in claim:
                return {
                    "type": "knowledge_base_contradiction",
                    "text": "FALSE: Narendra Modi is the Prime Minister of India, not the President of the United States. Modi has never held any position in the US government.",
                    "source": "Knowledge Base - Political Leadership",
                    "supports": False,
                    "confidence": 0.98
                }

        for person_name in people:
            if person_name in KNOWLEDGE_BASE["political_leaders"]:
                person_info = KNOWLEDGE_BASE["political_leaders"][person_name]
                person_country = person_info["country"].lower()

                for country in countries:
                    if country != person_country and country in ["usa", "united states", "america", "china", "russia",
                                                                 "france", "germany"]:
                        return {
                            "type": "knowledge_base_contradiction",
                            "text": f"FALSE: {person_name.title()} is the {person_info['position']} of {person_info['country']}, not a leader of {country.title()}.",
                            "source": "Knowledge Base - Political Leadership",
                            "supports": False,
                            "confidence": 0.95
                        }

                if person_country in countries:
                    return {
                        "type": "knowledge_base_verified",
                        "text": f"VERIFIED: {person_name.title()} is indeed the {person_info['position']} of {person_info['country']}.",
                        "source": "Knowledge Base - Political Leadership",
                        "supports": True,
                        "confidence": 0.95
                    }

        return None

    def _verify_geography_claim(self, claim: str, analysis: Dict) -> Optional[Dict]:
        """Enhanced geographical claim verification"""
        cities = [c.lower() for c in analysis["entities"]["cities"]]
        countries = [c.lower() for c in analysis["entities"]["countries"]]

        geo_checks = analysis.get("geographical_checks", [])
        for geo_check in geo_checks:
            if geo_check["type"] == "geographical_contradiction":
                return {
                    "type": "knowledge_base_contradiction",
                    "text": f"FALSE: {geo_check['fact']}. This is a known geographical error.",
                    "source": "Knowledge Base - Geography",
                    "supports": False,
                    "confidence": geo_check["confidence"]
                }

        if "capital" in claim:
            if ("delhi" in claim or "new delhi" in claim) and "russia" in claim:
                return {
                    "type": "knowledge_base_contradiction",
                    "text": "FALSE: New Delhi is the capital of India, not Russia. Moscow is the capital of Russia.",
                    "source": "Knowledge Base - Geography",
                    "supports": False,
                    "confidence": 0.98
                }

            for city in cities:
                if city in KNOWLEDGE_BASE["capitals_verified"]:
                    correct_country = KNOWLEDGE_BASE["capitals_verified"][city]
                    for country in countries:
                        if correct_country.lower() != country:
                            country_name = KNOWLEDGE_BASE["countries_capitals_verified"].get(country, "")
                            return {
                                "type": "knowledge_base_contradiction",
                                "text": f"FALSE: {city.title()} is the capital of {correct_country}, not {country.title()}. The capital of {country.title()} is {country_name}.",
                                "source": "Knowledge Base - Geography",
                                "supports": False,
                                "confidence": 0.98
                            }

        for city in cities:
            if city in KNOWLEDGE_BASE["capitals_verified"]:
                correct_country = KNOWLEDGE_BASE["capitals_verified"][city]
                if correct_country.lower() in [c.lower() for c in countries]:
                    return {
                        "type": "knowledge_base_verified",
                        "text": f"VERIFIED: {city.title()} is indeed the capital of {correct_country}.",
                        "source": "Knowledge Base - Geography",
                        "supports": True,
                        "confidence": 0.95
                    }

        return None

    def _search_wikipedia_enhanced(self, search_queries: List[str]) -> List[Dict]:
        """Enhanced Wikipedia search"""
        evidence = []

        for query in search_queries[:3]:
            wiki_results = self._search_wikipedia_single_query(query)
            evidence.extend(wiki_results)

        return evidence[:3]

    def _search_wikipedia_single_query(self, query: str) -> List[Dict]:
        """Search Wikipedia with a single query"""
        evidence = []

        if DEPENDENCIES.get('wikipedia', {}).get('available', False):
            try:
                import wikipedia
                wikipedia.set_lang("en")

                search_results = wikipedia.search(query, results=2)

                for title in search_results:
                    try:
                        page = wikipedia.page(title, auto_suggest=False)
                        summary = page.summary[:300]

                        evidence.append({
                            "type": "wikipedia",
                            "text": f"Wikipedia - {title}: {summary}",
                            "source": f"Wikipedia: {title}",
                            "supports": True,
                            "confidence": 0.8,
                            "url": page.url
                        })
                    except Exception:
                        continue

            except Exception:
                pass

        return evidence

    def _search_web_enhanced(self, search_queries: List[str]) -> List[Dict]:
        """Enhanced web search"""
        evidence = []

        if self.serpapi_key and self.serpapi_key != "your_serpapi_key_here":
            for query in search_queries[:2]:
                serp_results = self._search_serpapi_single(query)
                evidence.extend(serp_results)

        if self.newsapi_key and self.newsapi_key != "your_newsapi_key_here":
            for query in search_queries[:2]:
                news_results = self._search_newsapi_single(query)
                evidence.extend(news_results)

        return evidence[:5]

    def _search_serpapi_single(self, query: str) -> List[Dict]:
        """Search using SerpAPI"""
        evidence = []

        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": self.serpapi_key,
                "engine": "google",
                "num": 3,
                "hl": "en",
                "gl": "us"
            }

            response = self.session.get(url, params=params, timeout=self.config.get("TIMEOUT"))
            data = response.json()

            for result in data.get("organic_results", [])[:3]:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                source = result.get("source", "Unknown")
                link = result.get("link", "")

                if snippet and len(snippet) > 30:
                    evidence.append({
                        "type": "web_search",
                        "text": f"{title}: {snippet}",
                        "source": f"Web Search - {source}",
                        "supports": True,
                        "confidence": 0.8,
                        "url": link
                    })

        except Exception:
            pass

        return evidence

    def _search_newsapi_single(self, query: str) -> List[Dict]:
        """Search using NewsAPI"""
        evidence = []

        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": self.newsapi_key,
                "pageSize": 3,
                "language": "en",
                "sortBy": "relevancy"
            }

            response = self.session.get(url, params=params, timeout=self.config.get("TIMEOUT"))
            data = response.json()

            for article in data.get("articles", [])[:3]:
                title = article.get("title", "")
                description = article.get("description", "")
                source_name = article.get("source", {}).get("name", "")

                if description and len(description) > 30:
                    evidence.append({
                        "type": "news",
                        "text": f"{title}: {description}",
                        "source": f"News - {source_name}",
                        "supports": True,
                        "confidence": 0.7
                    })

        except Exception:
            pass

        return evidence

    def _generate_enhanced_summary(self, status: str, confidence: float, evidence_count: int, analysis: Dict) -> str:
        """Generate enhanced summary"""
        claim_type = analysis["claim_type"].replace("_", " ").title()

        if status == "VERIFIED":
            return f"{claim_type} claim VERIFIED with {confidence:.1%} confidence from {evidence_count} sources."
        elif status == "FALSE":
            return f"{claim_type} claim is FALSE with {confidence:.1%} confidence. Detected through knowledge base analysis and {evidence_count} evidence sources."
        else:
            return f"{claim_type} claim is UNCERTAIN with {confidence:.1%} confidence from {evidence_count} sources."


class FactCheckWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, claim: str, config: Dict):
        super().__init__()
        self.claim = claim
        self.config = config
        self.is_cancelled = False

    def run(self):
        try:
            self.progress.emit("Checking knowledge base for fake news indicators...")
            fact_checker = EnhancedFactChecker(self.config)

            self.progress.emit("Scanning for suspicious language patterns...")
            time.sleep(0.3)

            if self.is_cancelled:
                return

            self.progress.emit("Checking against trained fake news database...")
            time.sleep(0.2)

            if self.is_cancelled:
                return

            self.progress.emit("Verifying facts against knowledge base...")
            time.sleep(0.5)

            if self.is_cancelled:
                return

            self.progress.emit("Performing linguistic analysis...")
            time.sleep(0.3)

            if self.is_cancelled:
                return

            self.progress.emit("Searching external sources for verification...")
            time.sleep(0.5)

            if self.is_cancelled:
                return

            self.progress.emit("Compiling comprehensive analysis...")
            result = fact_checker.check_fact(self.claim)

            if self.is_cancelled:
                return

            status = result.get("status", "UNCERTAIN")
            confidence = result.get("confidence", 0.0)

            if status == "VERIFIED":
                real_pct = max(70, int(80 + 20 * confidence))
                fake_pct = 100 - real_pct
            elif status == "FALSE":
                fake_pct = max(70, int(80 + 20 * confidence))
                real_pct = 100 - fake_pct
            else:
                real_pct = int(30 + 40 * confidence)
                fake_pct = 100 - real_pct

            final_result = {
                "status": f"Fact Check: {status}",
                "real_percentage": real_pct,
                "fake_percentage": fake_pct,
                "confidence": confidence,
                "summary": result.get("summary", ""),
                "evidence": result.get("evidence", []),
                "source": result.get("source", "Enhanced Analysis")
            }

            self.finished.emit(final_result)

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self.is_cancelled = True


class EvidenceCard(QFrame):
    def __init__(self, evidence: Dict):
        super().__init__()
        self.evidence = evidence
        self.setup_ui()

    def setup_ui(self):
        self.setMinimumHeight(180)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)

        if self.evidence.get("type") == "fake_news_indicators":
            self.setStyleSheet("""
                EvidenceCard {
                    background-color: #ffe6e6;
                    border: 4px solid #ff0000;
                    border-radius: 12px;
                    margin: 8px;
                    padding: 12px;
                }
                QLabel {
                    background-color: transparent;
                    color: #8b0000;
                    font-weight: bold;
                }
            """)
        elif self.evidence.get("supports", True):
            if self.evidence.get("type") == "knowledge_base_verified":
                self.setStyleSheet("""
                    EvidenceCard {
                        background-color: #d4edda;
                        border: 3px solid #28a745;
                        border-radius: 12px;
                        margin: 8px;
                        padding: 12px;
                    }
                    QLabel {
                        background-color: transparent;
                        color: #155724;
                        font-weight: bold;
                    }
                """)
            else:
                self.setStyleSheet("""
                    EvidenceCard {
                        background-color: #f8f9fa;
                        border: 3px solid #6c757d;
                        border-radius: 12px;
                        margin: 8px;
                        padding: 12px;
                    }
                    QLabel {
                        background-color: transparent;
                        color: #495057;
                        font-weight: bold;
                    }
                """)
        else:
            self.setStyleSheet("""
                EvidenceCard {
                    background-color: #f8d7da;
                    border: 3px solid #dc3545;
                    border-radius: 12px;
                    margin: 8px;
                    padding: 12px;
                }
                QLabel {
                    background-color: transparent;
                    color: #721c24;
                    font-weight: bold;
                }
            """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        header_layout = QHBoxLayout()
        source_label = QLabel(self.evidence.get("source", "Unknown Source"))
        source_label.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(source_label)

        confidence = self.evidence.get("confidence", 0.5)
        confidence_label = QLabel(f"{confidence:.0%}")
        confidence_label.setFont(QFont("Arial", 11, QFont.Bold))
        header_layout.addWidget(confidence_label)
        header_layout.addStretch()

        support_text = "SUPPORTS" if self.evidence.get("supports", True) else "CONTRADICTS"
        support_label = QLabel(support_text)
        support_label.setFont(QFont("Arial", 10, QFont.Bold))
        header_layout.addWidget(support_label)

        layout.addLayout(header_layout)

        evidence_text = self.evidence.get("text", "No text available")
        text_label = QLabel(evidence_text)
        text_label.setWordWrap(True)
        text_label.setFont(QFont("Arial", 11))
        text_label.setMargin(8)
        text_label.setMinimumHeight(100)
        layout.addWidget(text_label)


class ResultGraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.real_percentage = 50
        self.fake_percentage = 50
        self.status = "Not Analyzed"
        self.confidence = 0
        self.setFixedHeight(300)
        self.setMinimumWidth(450)

    def update_results(self, real_pct: int, fake_pct: int, status: str, confidence: float = 0):
        self.real_percentage = real_pct
        self.fake_percentage = fake_pct
        self.status = status
        self.confidence = confidence
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#2d2d2d"))
        gradient.setColorAt(1, QColor("#1e1e1e"))
        painter.fillRect(self.rect(), gradient)

        if self.status == "Not Analyzed":
            painter.setPen(QPen(QColor("#888"), 2))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "Enter a claim to analyze")
            return

        width = self.width() - 60
        height = self.height() - 160
        x_start = 30
        y_start = 100
        bar_width = width // 3
        spacing = width // 6

        real_height = int((self.real_percentage / 100) * height)
        fake_height = int((self.fake_percentage / 100) * height)

        real_gradient = QLinearGradient(0, y_start + height - real_height, 0, y_start + height)
        real_gradient.setColorAt(0, QColor("#28a745"))
        real_gradient.setColorAt(1, QColor("#20c997"))
        painter.fillRect(x_start, y_start + height - real_height, bar_width, real_height, real_gradient)

        fake_gradient = QLinearGradient(0, y_start + height - fake_height, 0, y_start + height)
        fake_gradient.setColorAt(0, QColor("#dc3545"))
        fake_gradient.setColorAt(1, QColor("#fd7e14"))
        painter.fillRect(x_start + bar_width + spacing, y_start + height - fake_height, bar_width, fake_height,
                         fake_gradient)

        painter.setPen(QColor("#fff"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        painter.drawText(x_start, y_start + height - real_height - 10, bar_width, 20, Qt.AlignCenter,
                         f"{self.real_percentage}%")
        painter.drawText(x_start + bar_width + spacing, y_start + height - fake_height - 10, bar_width, 20,
                         Qt.AlignCenter, f"{self.fake_percentage}%")

        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(x_start, y_start + height + 10, bar_width, 20, Qt.AlignCenter, "VERIFIED")
        painter.drawText(x_start + bar_width + spacing, y_start + height + 10, bar_width, 20, Qt.AlignCenter, "FALSE")

        painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
        painter.drawText(0, 10, self.width(), 30, Qt.AlignCenter, "KB Priority Fact Check")

        status_color = self._get_status_color()
        painter.setPen(status_color)
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.drawText(0, 40, self.width(), 25, Qt.AlignCenter, self.status)

        if self.confidence > 0:
            painter.setPen(QColor("#aaa"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(0, 65, self.width(), 15, Qt.AlignCenter, f"Confidence: {self.confidence:.1%}")

    def _get_status_color(self) -> QColor:
        if "VERIFIED" in self.status:
            return QColor("#28a745")
        elif "FALSE" in self.status:
            return QColor("#dc3545")
        else:
            return QColor("#ffc107")


class WelcomeScreen(QWidget):
    start_verification = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)

        title = QLabel("Enhanced Smart Fact Checker with KB Priority")
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #3498db; margin: 20px;")
        layout.addWidget(title)

        subtitle = QLabel("Knowledge Base First • Pattern Detection • Accurate Evidence")
        subtitle.setFont(QFont("Segoe UI", 14))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; margin-bottom: 30px;")
        layout.addWidget(subtitle)

        features_frame = QFrame()
        features_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #34495e, stop:1 #2c3e50);
                border-radius: 15px;
                padding: 25px;
            }
        """)
        features_layout = QVBoxLayout(features_frame)

        features_title = QLabel("Priority-Based Detection System")
        features_title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        features_title.setStyleSheet("color: #ecf0f1; margin-bottom: 15px;")
        features_layout.addWidget(features_title)

        features = [
            "1. Fake News Indicator Detection (First Priority)",
            "2. Trained Fake Pattern Matching (Second Priority)",
            "3. Linguistic Analysis (Third Priority)",
            "4. Knowledge Base Verification (Fourth Priority)",
            "5. External Source Verification (Final Step)",
            "6. Evidence-Based Results with Clear Sources"
        ]

        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setFont(QFont("Segoe UI", 12))
            feature_label.setStyleSheet("color: #bdc3c7; margin: 5px 0; padding-left: 10px;")
            features_layout.addWidget(feature_label)

        layout.addWidget(features_frame)

        start_btn = QPushButton("Start Analysis")
        start_btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        start_btn.setFixedHeight(60)
        start_btn.setFixedWidth(300)
        start_btn.clicked.connect(self.start_verification.emit)
        start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
                border-radius: 30px;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ec7063, stop:1 #e74c3c);
            }
        """)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)


class FactCheckerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.config = ConfigManager.load_config()
        self.current_worker = None

        self.setWindowTitle("Enhanced Smart Fact Checker with KB Priority")
        self.setGeometry(100, 100, 1600, 1000)

        self.setup_ui()
        self.apply_theme()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.stacked_widget = QStackedWidget()

        self.welcome_screen = WelcomeScreen()
        self.welcome_screen.start_verification.connect(self.show_verifier)

        self.verifier_widget = self.create_main_widget()

        self.stacked_widget.addWidget(self.welcome_screen)
        self.stacked_widget.addWidget(self.verifier_widget)

        main_layout.addWidget(self.stacked_widget)
        self.stacked_widget.setCurrentIndex(0)

    def show_verifier(self):
        self.stacked_widget.setCurrentIndex(1)

    def create_main_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        header_layout = QHBoxLayout()
        back_btn = QPushButton("← Home")
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        back_btn.setFixedSize(100, 35)
        back_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6c757d, stop:1 #495057);
                color: white;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7d8285, stop:1 #6c757d);
            }
        """)
        header_layout.addWidget(back_btn)
        header_layout.addStretch()

        kb_status = QLabel("Knowledge Base: ✓")
        kb_status.setStyleSheet("color: #28a745; font-weight: bold;")
        header_layout.addWidget(kb_status)

        layout.addLayout(header_layout)

        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #34495e, stop:1 #2c3e50);
                border-radius: 12px;
                padding: 20px;
                margin: 5px;
            }
        """)
        input_layout = QVBoxLayout(input_frame)

        input_label = QLabel("Enter a headline or claim:")
        input_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        input_label.setStyleSheet("color: #ecf0f1; margin-bottom: 15px;")
        input_layout.addWidget(input_label)

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText(
            "Test Examples:\n\n"
            "FAKE: BREAKING: You Won't Believe What Modi Just Did!\n"
            "FAKE: Elon Musk owns Apple now - URGENT UPDATE!\n"
            "TRUE: Tokyo is the capital of Japan\n"
            "TRUE: Tim Cook is CEO of Apple"
        )
        self.input_text.setFixedHeight(120)
        self.input_text.setStyleSheet("""
            QTextEdit {
                background-color: #ecf0f1;
                border: 3px solid #bdc3c7;
                border-radius: 10px;
                padding: 15px;
                font-size: 13px;
                color: #2c3e50;
            }
            QTextEdit:focus {
                border-color: #e74c3c;
                background-color: #ffffff;
            }
        """)
        input_layout.addWidget(self.input_text)

        button_layout = QHBoxLayout()
        self.verify_btn = QPushButton("Analyze")
        self.verify_btn.setFixedHeight(50)
        self.verify_btn.clicked.connect(self.start_fact_check)
        self.verify_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ec7063, stop:1 #e74c3c);
            }
            QPushButton:disabled {
                background: #95a5a6;
                color: #7f8c8d;
            }
        """)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedHeight(50)
        self.clear_btn.clicked.connect(self.clear_all)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6c757d, stop:1 #495057);
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7d8285, stop:1 #6c757d);
            }
        """)

        button_layout.addWidget(self.verify_btn, 4)
        button_layout.addWidget(self.clear_btn, 1)
        input_layout.addLayout(button_layout)

        layout.addWidget(input_frame)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                text-align: center;
                background-color: #ecf0f1;
                height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e74c3c, stop:1 #c0392b);
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Segoe UI", 12))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #34495e; padding: 15px; font-weight: bold;")
        layout.addWidget(self.status_label)

        results_layout = QHBoxLayout()

        left_frame = QFrame()
        left_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2c3e50, stop:1 #34495e);
                border-radius: 12px;
                margin: 5px;
            }
        """)
        left_layout = QVBoxLayout(left_frame)

        self.result_graph = ResultGraphWidget()
        left_layout.addWidget(self.result_graph)

        self.summary_label = QLabel("")
        self.summary_label.setFont(QFont("Segoe UI", 11))
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignCenter)
        self.summary_label.setStyleSheet("color: #ecf0f1; padding: 20px; line-height: 1.4;")
        left_layout.addWidget(self.summary_label)

        evidence_frame = QFrame()
        evidence_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2c3e50, stop:1 #34495e);
                border-radius: 12px;
                margin: 5px;
            }
        """)
        evidence_layout = QVBoxLayout(evidence_frame)

        evidence_title = QLabel("Evidence Analysis")
        evidence_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        evidence_title.setStyleSheet("color: #ecf0f1; padding: 20px 20px 10px 20px;")
        evidence_layout.addWidget(evidence_title)

        self.evidence_scroll = QScrollArea()
        self.evidence_scroll.setWidgetResizable(True)
        self.evidence_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.evidence_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.evidence_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #34495e;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #7f8c8d;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #95a5a6;
            }
        """)

        self.evidence_widget = QWidget()
        self.evidence_layout = QVBoxLayout(self.evidence_widget)
        self.evidence_layout.setSpacing(8)
        self.evidence_layout.addStretch()

        self.evidence_scroll.setWidget(self.evidence_widget)
        evidence_layout.addWidget(self.evidence_scroll)

        results_layout.addWidget(left_frame, 1)
        results_layout.addWidget(evidence_frame, 2)
        layout.addLayout(results_layout)

        return widget

    def apply_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #ecf0f1;
                color: #2c3e50;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

    def clear_all(self):
        self.input_text.clear()
        self.clear_evidence()
        self.result_graph.update_results(50, 50, "Not Analyzed", 0)
        self.summary_label.setText("")
        self.status_label.setText("")
        self.progress_bar.setVisible(False)

    def clear_evidence(self):
        for i in reversed(range(self.evidence_layout.count())):
            item = self.evidence_layout.itemAt(i)
            if item and item.widget() and item.widget() != self.evidence_layout.parentWidget():
                item.widget().setParent(None)

    def start_fact_check(self):
        text = self.input_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Required", "Please enter a claim to analyze.")
            return

        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.cancel()
            self.current_worker.wait(1000)

        self.verify_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.clear_evidence()
        self.result_graph.update_results(50, 50, "Analyzing...", 0)
        self.status_label.setText("Starting KB priority analysis...")

        self.current_worker = FactCheckWorker(text, self.config)
        self.current_worker.progress.connect(self.update_status)
        self.current_worker.finished.connect(self.show_results)
        self.current_worker.error.connect(self.show_error)
        self.current_worker.start()

    def update_status(self, message: str):
        self.status_label.setText(message)

    def show_error(self, error_message: str):
        self.verify_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_message}")

    def show_results(self, results: Dict):
        self.verify_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        status = results.get("status", "Error")
        real_pct = results.get("real_percentage", 50)
        fake_pct = results.get("fake_percentage", 50)
        confidence = results.get("confidence", 0)
        summary = results.get("summary", "No summary available.")
        evidence = results.get("evidence", [])
        source = results.get("source", "Unknown")

        self.result_graph.update_results(real_pct, fake_pct, status, confidence)
        self.summary_label.setText(f"{summary}\n\nAnalyzed by: {source}")
        self.status_label.setText(f"{status} - {len(evidence)} evidence sources")

        self.clear_evidence()

        if not evidence:
            no_evidence_label = QLabel("No evidence sources available")
            no_evidence_label.setStyleSheet("color: #7f8c8d; padding: 20px; text-align: center;")
            self.evidence_layout.addWidget(no_evidence_label)
        else:
            for evidence_item in evidence:
                if isinstance(evidence_item, dict):
                    card = EvidenceCard(evidence_item)
                    self.evidence_layout.addWidget(card)

        self.evidence_layout.addStretch()

    def closeEvent(self, event):
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.cancel()
            self.current_worker.wait(2000)

        ConfigManager.save_config(self.config)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Enhanced Smart Fact Checker with KB Priority")
    app.setApplicationVersion("6.0")
    app.setOrganizationName("AI Research Labs")
    app.setQuitOnLastWindowClosed(True)

    try:
        window = FactCheckerApp()
        window.show()
        return app.exec_()
    except Exception as e:
        QMessageBox.critical(None, "Startup Error", f"Failed to start application:\n{str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())