webpackJsonp([0],{20:function(module,exports,a){var e,n;e=[a,exports,module],void 0!==(n=function(require,exports,module){"use strict";function generateTranslationFunction(n){var t=n||{},i=function f(n,i){var o=t[n]||n;return"object"===(void 0===i?"undefined":a(i))&&i?o.replace(e,function(a,e){var n=i[e],o=void 0===n?e:n;return t[n]||o}):o};return i.dictionary=t,i.merge=function(){var a=Array.prototype.slice.call(arguments,0),e,n,o;for(e=0;e<a.length;e+=1)if(n=a[e]&&a[e].dictionary)for(o in n)o in t||(t[o]=n[o]);return i},i}var a="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(a){return typeof a}:function(a){return a&&"function"==typeof Symbol&&a.constructor===Symbol&&a!==Symbol.prototype?"symbol":typeof a},e=/[#!]\{([^}]+?)\}/g;module.exports=generateTranslationFunction}.apply(exports,e))&&(module.exports=n)},49:function(module,exports,a){var e,n;e=[a,exports,module,a(2),a(40),a(62)],void 0!==(n=function(require,exports,module){"use strict";function toIetfLanguageTag(a){return(a||"").replace(/_/g,"-").split(/[;=,]/)[0].toLowerCase().replace(/(-[a-z]{1,}$)/,function(a){return a.toUpperCase()})}function getIetfLanguageTag(){try{return toIetfLanguageTag(n.getLocale())}catch(a){return e.push(["user.language.error",{error:a}]),"en-US"}}function toLanguageCode(a){return toIetfLanguageTag(a).split("-")[0]}function getLanguageCode(){return toLanguageCode(getIetfLanguageTag())}function getMomentLanguage(){return getIetfLanguageTag().toLowerCase()}function getFacebookLocaleString(){var a=getIetfLanguageTag().replace("-","_"),e=a.split("_")[0];return u[e]||a}function languageCodeToName(a){var e=toIetfLanguageTag(a),n=e.split("-"),i;return _.any(n,function(a,e){var o=n.slice(0,n.length-e).join("-");return!!(i=t[toIetfLanguageTag(o)])})&&null!=i?i:a}function languageCodeCSVtoLanguages(a){var e=_.compact((a||"").split(/,\s*/g));return _.reduce(e,function(a,e){return a[e]=languageCodeToName(e),a},{})}function isRightToLeft(a){return _(["ar","he"]).contains(toLanguageCode(a))}function latinizeText(a){return a.replace(/[^A-Za-z0-9\[\] ]/g,function(a){return s[a]||a})}function getMobileBadgeLanguageCode(){var a=["zh-cn","zh-tw","fr","en","pt-br","pt-pt","es","tr","ru"],e=getIetfLanguageTag().toLowerCase();if(_(a).contains(e))return e;var n=getLanguageCode();return"pt"===n?"pt-br":"es"===n?"es":"zh"===n?"zh-cn":"en"}var _=a(2),e=a(40),n=a(62),t={ab:"Abkhaz",aa:"Afar",af:"Afrikaans",ak:"Akan",sq:"Albanian",am:"Amharic",ar:"Arabic",an:"Aragonese",hy:"Armenian",as:"Assamese",av:"Avaric",ae:"Avestan",ay:"Aymara",az:"Azerbaijani",bm:"Bambara",ba:"Bashkir",eu:"Basque",be:"Belarusian",bn:"Bengali",bh:"Bihari",bi:"Bislama",bs:"Bosnian",br:"Breton",bg:"Bulgarian",my:"Burmese",ca:"Catalan",ch:"Chamorro",ce:"Chechen",ny:"Chichewa",zh:"Chinese","zh-CN":"Chinese (Simplified)","zh-TW":"Chinese (Traditional)",cv:"Chuvash",kw:"Cornish",co:"Corsican",cr:"Cree",hr:"Croatian",cs:"Czech",da:"Danish",dv:"Divehi",nl:"Dutch",dz:"Dzongkha",en:"English",eo:"Esperanto",et:"Estonian",ee:"Ewe",fo:"Faroese",fj:"Fijian",fi:"Finnish",fr:"French",ff:"Fula",gl:"Galician",ka:"Georgian",de:"German",el:"Greek",gn:"Guaraní",gu:"Gujarati",ht:"Haitian",ha:"Hausa",he:"Hebrew",hz:"Herero",hi:"Hindi",ho:"Hiri Motu",hu:"Hungarian",ia:"Interlingua",id:"Indonesian",ie:"Interlingue",ga:"Irish",ig:"Igbo",ik:"Inupiaq",io:"Ido",is:"Icelandic",it:"Italian",iu:"Inuktitut",ja:"Japanese",jv:"Javanese",kl:"Kalaallisut",kn:"Kannada",kr:"Kanuri",ks:"Kashmiri",kk:"Kazakh",km:"Khmer",ki:"Kikuyu",rw:"Kinyarwanda",ky:"Kyrgyz",kv:"Komi",kg:"Kongo",ko:"Korean",ku:"Kurdish",kj:"Kwanyama",la:"Latin",lb:"Luxembourgish",lg:"Ganda",li:"Limburgish",ln:"Lingala",lo:"Lao",lt:"Lithuanian",lu:"Luba-Katanga",lv:"Latvian",gv:"Manx",mk:"Macedonian (FYROM)",mg:"Malagasy",ms:"Malay",ml:"Malayalam",mt:"Maltese",mi:"Māori",mr:"Marathi",mh:"Marshallese",mn:"Mongolian",na:"Nauru",nv:"Navajo",nb:"Norwegian Bokmål",nd:"North Ndebele",ne:"Nepali",ng:"Ndonga",nn:"Norwegian Nynorsk",no:"Norwegian",ii:"Nuosu",nr:"South Ndebele",oc:"Occitan",oj:"Ojibwe",cu:"Old Church Slavonic",om:"Oromo",or:"Oriya",os:"Ossetian",pa:"Panjabi",pi:"Pāli",fa:"Persian",pl:"Polish",ps:"Pashto",pt:"Portuguese (Brazilian)","pt-BR":"Portuguese (Brazilian)","pt-PT":"Portuguese (European)",qu:"Quechua",rm:"Romansh",rn:"Kirundi",ro:"Romanian",ru:"Russian",sa:"Sanskrit",sc:"Sardinian",sd:"Sindhi",se:"Northern Sami",sm:"Samoan",sg:"Sango",sr:"Serbian",gd:"Gaelic",sn:"Shona",si:"Sinhala",sk:"Slovak",sl:"Slovene",so:"Somali",st:"Southern Sotho",es:"Spanish",su:"Sundanese",sw:"Swahili",ss:"Swati",sv:"Swedish",ta:"Tamil",te:"Telugu",tg:"Tajik",th:"Thai",ti:"Tigrinya",bo:"Tibetan",tk:"Turkmen",tl:"Tagalog",tn:"Tswana",to:"Tonga",tr:"Turkish",ts:"Tsonga",tt:"Tatar",tw:"Twi",ty:"Tahitian",ug:"Uighur",uk:"Ukrainian",ur:"Urdu",uz:"Uzbek",ve:"Venda",vi:"Vietnamese",vo:"Volapük",wa:"Walloon",cy:"Welsh",wo:"Wolof",fy:"Western Frisian",xh:"Xhosa",yi:"Yiddish",yo:"Yoruba",za:"Zhuang",zu:"Zulu"},i=["ar","ca","de","en","es","fr","he","hi","it","ja","ko","pt-BR","pt-PT","ru","tr","zh-CN","zh-TW"],o=["am","ar","az","bg","bn","bs","ca","cs","da","de","el","es","et","eu","en","fa","fi","tl","fr","he","hi","hr","hu","hy","id","it","ja","ka","kk","km","kn","ko","lt","lv","mk","mn","mr","ms","my","ne","nl","no","pl","ps","pt-BR","pt-PT","ro","ru","rw","sk","sl","sq","sr","sv","sw","ta","te","th","tr","uk","ur","uz","vi","yo","zh-CN","zh-TW"],r=["en","es","fr","pt","ru","tr","zh","zh-tw"],u={ar:"ar_AR",es:"es_LA"},g={A:"Á Ă Ắ Ặ Ằ Ẳ Ẵ Ǎ Â Ấ Ậ Ầ Ẩ Ẫ Ä Ǟ Ȧ Ǡ Ạ Ȁ À Ả Ȃ Ā Ą Å Ǻ Ḁ Ⱥ Ã Ɐ ᴀ",AA:"Ꜳ",AE:"Æ Ǽ Ǣ ᴁ",AO:"Ꜵ",AU:"Ꜷ",AV:"Ꜹ Ꜻ",AY:"Ꜽ",B:"Ḃ Ḅ Ɓ Ḇ Ƀ Ƃ ʙ ᴃ",C:"Ć Č Ç Ḉ Ĉ Ċ Ƈ Ȼ Ꜿ ᴄ",D:"Ď Ḑ Ḓ Ḋ Ḍ Ɗ Ḏ ǲ ǅ Đ Ƌ Ꝺ ᴅ",DZ:"Ǳ Ǆ",E:"É Ĕ Ě Ȩ Ḝ Ê Ế Ệ Ề Ể Ễ Ḙ Ë Ė Ẹ Ȅ È Ẻ Ȇ Ē Ḗ Ḕ Ę Ɇ Ẽ Ḛ Ɛ Ǝ ᴇ ⱻ",ET:"Ꝫ",F:"Ḟ Ƒ Ꝼ ꜰ",G:"Ǵ Ğ Ǧ Ģ Ĝ Ġ Ɠ Ḡ Ǥ Ᵹ ɢ ʛ",H:"Ḫ Ȟ Ḩ Ĥ Ⱨ Ḧ Ḣ Ḥ Ħ ʜ",I:"Í Ĭ Ǐ Î Ï Ḯ İ Ị Ȉ Ì Ỉ Ȋ Ī Į Ɨ Ĩ Ḭ ɪ",R:"Ꞃ Ŕ Ř Ŗ Ṙ Ṛ Ṝ Ȑ Ȓ Ṟ Ɍ Ɽ ʁ ʀ ᴙ ᴚ",S:"Ꞅ Ś Ṥ Š Ṧ Ş Ŝ Ș Ṡ Ṣ Ṩ ꜱ",T:"Ꞇ Ť Ţ Ṱ Ț Ⱦ Ṫ Ṭ Ƭ Ṯ Ʈ Ŧ ᴛ",IS:"Ꝭ",J:"Ĵ Ɉ ᴊ",K:"Ḱ Ǩ Ķ Ⱪ Ꝃ Ḳ Ƙ Ḵ Ꝁ Ꝅ ᴋ",L:"Ĺ Ƚ Ľ Ļ Ḽ Ḷ Ḹ Ⱡ Ꝉ Ḻ Ŀ Ɫ ǈ Ł Ꞁ ʟ ᴌ",LJ:"Ǉ",M:"Ḿ Ṁ Ṃ Ɱ Ɯ ᴍ",N:"Ń Ň Ņ Ṋ Ṅ Ṇ Ǹ Ɲ Ṉ Ƞ ǋ Ñ ɴ ᴎ",NJ:"Ǌ",O:"Ó Ŏ Ǒ Ô Ố Ộ Ồ Ổ Ỗ Ö Ȫ Ȯ Ȱ Ọ Ő Ȍ Ò Ỏ Ơ Ớ Ợ Ờ Ở Ỡ Ȏ Ꝋ Ꝍ Ō Ṓ Ṑ Ɵ Ǫ Ǭ Ø Ǿ Õ Ṍ Ṏ Ȭ Ɔ ᴏ ᴐ",OI:"Ƣ",OO:"Ꝏ",OU:"Ȣ ᴕ",P:"Ṕ Ṗ Ꝓ Ƥ Ꝕ Ᵽ Ꝑ ᴘ",Q:"Ꝙ Ꝗ",V:"Ʌ Ꝟ Ṿ Ʋ Ṽ ᴠ",TZ:"Ꜩ",U:"Ú Ŭ Ǔ Û Ṷ Ü Ǘ Ǚ Ǜ Ǖ Ṳ Ụ Ű Ȕ Ù Ủ Ư Ứ Ự Ừ Ử Ữ Ȗ Ū Ṻ Ų Ů Ũ Ṹ Ṵ ᴜ",VY:"Ꝡ",W:"Ẃ Ŵ Ẅ Ẇ Ẉ Ẁ Ⱳ ᴡ",X:"Ẍ Ẋ",Y:"Ý Ŷ Ÿ Ẏ Ỵ Ỳ Ƴ Ỷ Ỿ Ȳ Ɏ Ỹ ʏ",Z:"Ź Ž Ẑ Ⱬ Ż Ẓ Ȥ Ẕ Ƶ ᴢ",IJ:"Ĳ",OE:"Œ ɶ",a:"á ă ắ ặ ằ ẳ ẵ ǎ â ấ ậ ầ ẩ ẫ ä ǟ ȧ ǡ ạ ȁ à ả ȃ ā ą ᶏ ẚ å ǻ ḁ ⱥ ã ɐ ₐ",aa:"ꜳ",ae:"æ ǽ ǣ ᴂ",ao:"ꜵ",au:"ꜷ",av:"ꜹ ꜻ",ay:"ꜽ",b:"ḃ ḅ ɓ ḇ ᵬ ᶀ ƀ ƃ",o:"ɵ ó ŏ ǒ ô ố ộ ồ ổ ỗ ö ȫ ȯ ȱ ọ ő ȍ ò ỏ ơ ớ ợ ờ ở ỡ ȏ ꝋ ꝍ ⱺ ō ṓ ṑ ǫ ǭ ø ǿ õ ṍ ṏ ȭ ɔ ᶗ ᴑ ᴓ ₒ",c:"ć č ç ḉ ĉ ɕ ċ ƈ ȼ ↄ ꜿ",d:"ď ḑ ḓ ȡ ḋ ḍ ɗ ᶑ ḏ ᵭ ᶁ đ ɖ ƌ ꝺ",i:"ı í ĭ ǐ î ï ḯ ị ȉ ì ỉ ȋ ī į ᶖ ɨ ĩ ḭ ᴉ ᵢ",j:"ȷ ɟ ʄ ǰ ĵ ʝ ɉ ⱼ",dz:"ǳ ǆ",e:"é ĕ ě ȩ ḝ ê ế ệ ề ể ễ ḙ ë ė ẹ ȅ è ẻ ȇ ē ḗ ḕ ⱸ ę ᶒ ɇ ẽ ḛ ɛ ᶓ ɘ ǝ ₑ",et:"ꝫ",f:"ḟ ƒ ᵮ ᶂ ꝼ",g:"ǵ ğ ǧ ģ ĝ ġ ɠ ḡ ᶃ ǥ ᵹ ɡ ᵷ",h:"ḫ ȟ ḩ ĥ ⱨ ḧ ḣ ḥ ɦ ẖ ħ ɥ ʮ ʯ",hv:"ƕ",r:"ꞃ ŕ ř ŗ ṙ ṛ ṝ ȑ ɾ ᵳ ȓ ṟ ɼ ᵲ ᶉ ɍ ɽ ɿ ɹ ɻ ɺ ⱹ ᵣ",s:"ꞅ ſ ẜ ẛ ẝ ś ṥ š ṧ ş ŝ ș ṡ ṣ ṩ ʂ ᵴ ᶊ ȿ",t:"ꞇ ť ţ ṱ ț ȶ ẗ ⱦ ṫ ṭ ƭ ṯ ᵵ ƫ ʈ ŧ ʇ",is:"ꝭ",k:"ḱ ǩ ķ ⱪ ꝃ ḳ ƙ ḵ ᶄ ꝁ ꝅ ʞ",l:"ĺ ƚ ɬ ľ ļ ḽ ȴ ḷ ḹ ⱡ ꝉ ḻ ŀ ɫ ᶅ ɭ ł ꞁ",lj:"ǉ",m:"ḿ ṁ ṃ ɱ ᵯ ᶆ ɯ ɰ",n:"ń ň ņ ṋ ȵ ṅ ṇ ǹ ɲ ṉ ƞ ᵰ ᶇ ɳ ñ",nj:"ǌ",oi:"ƣ",oo:"ꝏ",ou:"ȣ",p:"ṕ ṗ ꝓ ƥ ᵱ ᶈ ꝕ ᵽ ꝑ",q:"ꝙ ʠ ɋ ꝗ",u:"ᴝ ú ŭ ǔ û ṷ ü ǘ ǚ ǜ ǖ ṳ ụ ű ȕ ù ủ ư ứ ự ừ ử ữ ȗ ū ṻ ų ᶙ ů ũ ṹ ṵ ᵤ",th:"ᵺ",oe:"ᴔ œ",v:"ʌ ⱴ ꝟ ṿ ʋ ᶌ ⱱ ṽ ᵥ",w:"ʍ ẃ ŵ ẅ ẇ ẉ ẁ ⱳ ẘ",y:"ʎ ý ŷ ÿ ẏ ỵ ỳ ƴ ỷ ỿ ȳ ẙ ɏ ỹ",tz:"ꜩ",ue:"ᵫ",um:"ꝸ",vy:"ꝡ",x:"ẍ ẋ ᶍ ₓ",z:"ź ž ẑ ʑ ⱬ ż ẓ ȥ ẕ ᵶ ᶎ ʐ ƶ ɀ",ff:"ﬀ",ffi:"ﬃ",ffl:"ﬄ",fi:"ﬁ",fl:"ﬂ",ij:"ĳ",st:"ﬆ"},s=_.chain(g).map(function(a,e){return _.map(a.split(" "),function(a){return[a,e]})}).flatten(!0).object().value();module.exports={languages:t,languageCodeCSVtoLanguages:languageCodeCSVtoLanguages,languageCodeToName:languageCodeToName,latinizeText:latinizeText,isRightToLeft:isRightToLeft,coursePrimaryLanguageTags:i,courseSubtitleLanguageTags:o,getIetfLanguageTag:getIetfLanguageTag,getLanguageCode:getLanguageCode,getMomentLanguage:getMomentLanguage,getFacebookLocaleString:getFacebookLocaleString,toIetfLanguageTag:toIetfLanguageTag,toLanguageCode:toLanguageCode,getMobileBadgeLanguageCode:getMobileBadgeLanguageCode,supportedLanguageSubdomains:r}}.apply(exports,e))&&(module.exports=n)},62:function(module,exports,a){var e={},n=a(20),t=n(e);t.getLocale=function(){return"en"},module.exports=t}});