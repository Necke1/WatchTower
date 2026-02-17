#!pip install classla #our ml
#!pip install phunspell #for spellchecking
import classla
classla.download('sr')# download the ML model, specificalz for Serbian language

import phunspell
self.latin_dict = phunspell.Phunspell('sr-Latn') # type: ignore
self.cyrillic_dict = phunspell.Phunspell('sr') # type: ignore

