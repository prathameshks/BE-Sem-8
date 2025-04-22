import re

class MorphologyAnalyzer:
    def __init__(self):
        self.examples = {
            "happy": {
                "prefixes": [("un", "unhappy", "Derivational"), 
                            ("super", "superhappy", "Derivational")],
                "suffixes": [("ness", "happiness", "Derivational"), 
                            ("ly", "happily", "Derivational")]
            },
            "read": {
                "prefixes": [("re", "reread", "Derivational"), 
                            ("un", "unread", "Derivational")],
                "suffixes": [("er", "reader", "Derivational"), 
                            ("ing", "reading", "Inflectional")]
            },
            "write": {
                "prefixes": [("re", "rewrite", "Derivational"), 
                            ("under", "underwrite", "Derivational")],
                "suffixes": [("er", "writer", "Derivational"), 
                            ("ing", "writing", "Inflectional")]
            }
        }
        
    def analyze_word(self, word):
        word = word.strip().lower()
        if not word:
            return "Please enter a root word"
            
        if word in self.examples:
            analysis = f"Root word: {word}\n\n"
            analysis += "This word can take various prefixes and suffixes.\n"
            analysis += str(self.examples[word])
        else:
            analysis = f"Root word: {word}\n\n"
            analysis += "Try adding common prefixes (un-, re-, dis-, in-) and suffixes (-ed, -ing, -er, -ness, -ly).\n"
            analysis += "See how adding these changes the meaning or grammatical function of the word."
            
        return analysis
        
    def check_answers(self, word, affixes):
        word = word.strip().lower()
        if not word:
            return "Please enter a root word"
        
        for affix_type, affix, new_word, process in affixes:
            affix = affix.strip()
            new_word = new_word.strip().lower()
            
            if not affix and not new_word:
                continue
                
            if not affix or not new_word or not process:
                return f"Incomplete entry: {affix_type}, {affix}, {new_word}, {process}"
                
            if affix_type == "prefix":
                expected_word = affix + word
                if new_word != expected_word:
                    return f"Incorrect word with prefix '{affix}': expected '{expected_word}', got '{new_word}'"
            else:
                expected_word = word + affix
                if new_word != expected_word:
                    return f"Incorrect word with suffix '{affix}': expected '{expected_word}', got '{new_word}'"
        
        analysis = f"Morphological Analysis for '{word}':\n\n"
        for affix_type, affix, new_word, process in affixes:
            if not affix:
                continue
            analysis += f"• {affix_type} '{affix}' + '{word}' → '{new_word}' ({process})\n"
            if process == "Derivational":
                analysis += "  This changes the meaning or part of speech of the word.\n"
            elif process == "Inflectional":
                analysis += "  This expresses grammatical function without changing the core meaning.\n"
        return analysis

    def show_hint(self, word):
        word = word.strip().lower()
        if not word:
            return "Please enter a root word first"
            
        if word in self.examples:
            hint = f"Hints for '{word}':\n\n"
            hint += f"Common prefixes: un-, re-, in-, dis-\n"
            hint += f"Common suffixes: -er, -ing, -ness, -ly\n\n"
            
            if word == "happy":
                hint += "Try: un+happy, happy+ness"
            elif word == "read":
                hint += "Try: re+read, read+er, read+ing"
            elif word == "write":
                hint += "Try: re+write, write+er"
            
            return hint
        else:
            hint = f"Hints for general morphology:\n\n"
            hint += f"Common prefixes: un-, re-, in-, dis-\n"
            hint += f"Common suffixes: -er, -ing, -ness, -ly, -ment\n\n"
            hint += f"Remember:\n"
            hint += f"• Prefixes often change meaning (re- = again, un- = not)\n"
            hint += f"• Suffixes can change word class (verb → noun, adj → adverb)\n"
            
            return hint

# Example Usage:
if __name__ == "__main__":
    analyzer = MorphologyAnalyzer()
    
    word = "happier"
    print(analyzer.analyze_word(word))
    
    affixes = [
        ("prefix", "un", "unhappy", "Derivational"),
        ("prefix", "re", "", ""),
        ("suffix", "ness", "happiness", "Derivational"),
        ("suffix", "ly", "", "")
    ]
    
    print(analyzer.check_answers(word, affixes))
    
    print(analyzer.show_hint(word))