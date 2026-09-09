/* A fixture encyclopedia, and a deliberately naive search over it.
 *
 * Two reasons this is fixtures rather than the live API:
 *
 * 1. A benchmark that hits the network is not a benchmark. Wikipedia's ranking
 *    changes under you, and a regression suite whose expected answers move is
 *    worthless as a regression suite.
 * 2. The environment this was built in has no egress, so the alternative was
 *    not "test against Wikipedia", it was "do not test".
 *
 * The article prose below is written for this file. It is not copied from
 * Wikipedia -- which also means the copy-overlap metric measures the anti-copy
 * layer honestly: any long verbatim run in an answer came from this text, and
 * the layer is supposed to break it.
 *
 * The search function is intentionally naive: title-token and body-token
 * matching with a length prior, which is roughly what a lexical search engine
 * does and is *exactly* the behaviour that makes "High School High" a plausible
 * hit for "high school". Making the fixture search smarter would hide the
 * problem this whole change exists to fix.
 */

"use strict";

/* id, title, redirects (titles that resolve here), disambiguation, extract
   (the intro the search API returns), full (what the extracts API returns). */
const PAGES = [
  {
    id: 1001, title: "Secondary school",
    redirects: ["high school", "highschool", "senior high school", "high schools"],
    extract: "A secondary school is an institution that educates adolescents in the years " +
      "between primary education and higher education or employment.",
    full: "A secondary school is an institution that educates adolescents in the years " +
      "between primary education and higher education or employment. Attendance usually " +
      "begins somewhere between the ages of eleven and fourteen and continues for four to " +
      "seven years depending on the country. Teaching is organised by subject specialists " +
      "rather than by a single class teacher, which is one of the clearest breaks from " +
      "primary schooling. Most systems end the stage with an examination or a leaving " +
      "certificate that governs access to university or to vocational training. The term " +
      "high school is used for this stage in the United States, Canada, Australia and a " +
      "number of other countries. See also: education, curriculum, school."
  },
  {
    id: 1002, title: "High School High", redirects: [],
    extract: "High School High is a 1996 American comedy film directed by Hart Bochner and " +
      "starring Jon Lovitz as an idealistic teacher.",
    full: "High School High is a 1996 American comedy film directed by Hart Bochner and " +
      "starring Jon Lovitz as an idealistic teacher who transfers to a troubled inner-city " +
      "school. The film parodies the inspirational-teacher genre that had been popular in " +
      "the preceding decade. It was produced by the team behind several other spoof comedies " +
      "of the period and received largely negative reviews on release."
  },
  {
    id: 1003, title: "High School Musical",
    redirects: ["high school musical", "hsm"],
    extract: "High School Musical is a 2006 American musical television film and the first " +
      "instalment of a franchise of the same name.",
    full: "High School Musical is a 2006 American musical television film and the first " +
      "instalment of a franchise of the same name. It follows two students from different " +
      "social circles who audition for the leads in their school's production. The film was " +
      "an unexpected ratings success and spawned two sequels, a stage adaptation and a " +
      "streaming series."
  },
  {
    id: 1010, title: "Mathematics",
    redirects: ["math", "maths", "mathematic"],
    extract: "Mathematics is the study of quantity, structure, space and change, carried out " +
      "by stating assumptions precisely and deducing their consequences.",
    full: "Mathematics is the study of quantity, structure, space and change, carried out by " +
      "stating assumptions precisely and deducing their consequences. Its results are " +
      "established by proof rather than by observation, which distinguishes it from the " +
      "empirical sciences. The main branches include arithmetic, algebra, geometry, analysis " +
      "and probability, though the boundaries between them are not sharp. Mathematics is " +
      "used throughout the natural sciences, engineering, medicine and economics, and areas " +
      "developed for their own sake have repeatedly turned out to describe the physical world."
  },
  {
    id: 1011, title: "MathJax", redirects: [],
    extract: "MathJax is a JavaScript display engine that renders mathematical notation in " +
      "web browsers.",
    full: "MathJax is a JavaScript display engine that renders mathematical notation in web " +
      "browsers. It accepts input written in LaTeX, MathML or AsciiMath and produces output " +
      "using HTML with CSS, or SVG. The project is open source and is widely deployed on " +
      "academic publishing platforms and on wikis that carry mathematical content."
  },
  {
    id: 1012, title: "MathML", redirects: [],
    extract: "MathML is a markup language for describing mathematical notation and capturing " +
      "both its structure and its content.",
    full: "MathML is a markup language for describing mathematical notation and capturing " +
      "both its structure and its content. It is an application of XML and is maintained as " +
      "a web standard. Presentation MathML records how an expression should look, while " +
      "content MathML records what it means."
  },
  {
    id: 1013, title: "Math rock", redirects: ["math rock"],
    extract: "Math rock is a style of rock music characterised by irregular metres, angular " +
      "melodies and abrupt changes of rhythm.",
    full: "Math rock is a style of rock music characterised by irregular metres, angular " +
      "melodies and abrupt changes of rhythm. It emerged in the late 1980s, drawing on " +
      "progressive rock and on the noisier end of American indie music. Guitar parts are " +
      "often played in unusual tunings and interlock rather than double each other."
  },
  {
    id: 1020, title: "Science", redirects: ["science"],
    extract: "Science is a systematic enterprise that builds and organises knowledge about " +
      "the natural world in the form of testable explanations and predictions.",
    full: "Science is a systematic enterprise that builds and organises knowledge about the " +
      "natural world in the form of testable explanations and predictions. Its central " +
      "commitment is that a claim must be checkable against observation, and that a claim " +
      "which no observation could contradict is not a scientific one. Modern practice is " +
      "divided into the natural sciences, the social sciences and the formal sciences, with " +
      "engineering and medicine applying their results."
  },
  {
    id: 1030, title: "Artificial intelligence",
    redirects: ["ai", "a.i.", "artificial intelligence"],
    extract: "Artificial intelligence is the field concerned with building systems that " +
      "perform tasks normally associated with human reasoning, perception and language.",
    full: "Artificial intelligence is the field concerned with building systems that perform " +
      "tasks normally associated with human reasoning, perception and language. Early work " +
      "concentrated on symbolic search and hand-written rules; most current systems instead " +
      "fit statistical models to large collections of examples. Applications include machine " +
      "translation, speech recognition, medical image analysis and robotics."
  },
  {
    id: 1040, title: "Photosynthesis", redirects: ["photosynthesis"],
    extract: "Photosynthesis is the process by which plants, algae and some bacteria convert " +
      "light energy into chemical energy stored as sugars.",
    full: "Photosynthesis is the process by which plants, algae and some bacteria convert " +
      "light energy into chemical energy stored as sugars. Light is absorbed by chlorophyll " +
      "and other pigments held in the thylakoid membranes of chloroplasts. The light " +
      "reactions split water, release oxygen and produce ATP and NADPH; the Calvin cycle " +
      "then uses those products to fix carbon dioxide into carbohydrate. Almost all of the " +
      "oxygen in the atmosphere is a by-product of this process."
  },
  {
    id: 1050, title: "JavaScript", redirects: ["javascript", "js"],
    extract: "JavaScript is a programming language that is one of the core technologies of " +
      "the web, alongside HTML and CSS.",
    full: "JavaScript is a programming language that is one of the core technologies of the " +
      "web, alongside HTML and CSS. It is dynamically typed, supports first-class functions " +
      "and prototype-based objects, and is executed by an engine built into every major " +
      "browser. Server-side runtimes have since taken the same engines outside the browser. " +
      "Despite the name it is unrelated to Java."
  },
  {
    id: 1051, title: "Java (programming language)",
    redirects: ["java (programming language)"],
    extract: "Java is a class-based, object-oriented programming language designed so that " +
      "compiled code can run on any platform with a suitable virtual machine.",
    full: "Java is a class-based, object-oriented programming language designed so that " +
      "compiled code can run on any platform with a suitable virtual machine. Source is " +
      "compiled to bytecode, which the Java Virtual Machine executes. It is statically typed " +
      "and manages memory automatically. It is widely used for enterprise back ends and for " +
      "Android applications."
  },
  {
    id: 1052, title: "Java", redirects: ["java"], disambiguation: false,
    extract: "Java is an island of Indonesia and the most populous island in the world.",
    full: "Java is an island of Indonesia and the most populous island in the world, home to " +
      "more than half of the country's population. It is formed largely by volcanic activity " +
      "and runs roughly east to west south of Borneo. Jakarta, the national capital, lies on " +
      "its north-west coast. The island gives its name to a variety of coffee grown there."
  },
  {
    id: 1060, title: "Paris", redirects: ["paris"],
    extract: "Paris is the capital and most populous city of France.",
    full: "Paris is the capital and most populous city of France, situated on the river Seine " +
      "in the north of the country. It has been the political centre of France since the " +
      "tenth century and is a major centre for finance, diplomacy, fashion and the arts. The " +
      "capital of France is Paris, and the city's administrative region is Ile-de-France."
  },
  {
    id: 1061, title: "France", redirects: ["france"],
    extract: "France is a country in Western Europe whose capital is Paris.",
    full: "France is a country in Western Europe whose capital is Paris. Its territory also " +
      "includes overseas regions in the Caribbean, South America and the Indian Ocean. The " +
      "capital of France is Paris. It is a founding member of the European Union and a " +
      "permanent member of the United Nations Security Council.",
    facts: { capital: "Paris", population: "68 million", currency: "euro" }
  },
  {
    id: 1070, title: "Albert Einstein", redirects: ["albert einstein", "einstein"],
    extract: "Albert Einstein was a theoretical physicist who developed the theory of " +
      "relativity, one of the two pillars of modern physics.",
    full: "Albert Einstein was a theoretical physicist who developed the theory of " +
      "relativity, one of the two pillars of modern physics. He was born in Ulm in 1879 and " +
      "died in Princeton in 1955. His 1905 papers on the photoelectric effect, Brownian " +
      "motion and special relativity reshaped the field in a single year, and he received the " +
      "Nobel Prize in Physics in 1921 for the first of them. General relativity, published in " +
      "1915, described gravitation as the curvature of spacetime."
  },
  {
    id: 1080, title: "Python (programming language)",
    redirects: ["python (programming language)"],
    extract: "Python is a high-level, general-purpose programming language known for readable " +
      "syntax and a large standard library.",
    full: "Python is a high-level, general-purpose programming language known for readable " +
      "syntax and a large standard library. It is dynamically typed and manages memory " +
      "automatically, and it supports procedural, object-oriented and functional styles. It " +
      "is heavily used in scientific computing, data analysis, automation and web back ends."
  },
  {
    id: 1081, title: "Python", redirects: ["python"], disambiguation: true,
    extract: "Python may refer to the programming language, to snakes of the family " +
      "Pythonidae, or to the comedy group Monty Python.",
    full: "Python may refer to the programming language, to snakes of the family Pythonidae, " +
      "or to the comedy group Monty Python. See also: Python (mythology), Python (genus)."
  },
  {
    id: 1082, title: "Pythonidae", redirects: [],
    extract: "Pythonidae is a family of nonvenomous snakes found in Africa, Asia and " +
      "Australia.",
    full: "Pythonidae is a family of nonvenomous snakes found in Africa, Asia and Australia. " +
      "Members kill prey by constriction and include some of the longest snakes in the world."
  },
  {
    id: 1083, title: "Monty Python", redirects: [],
    extract: "Monty Python were a British comedy troupe formed in 1969, best known for the " +
      "television series Monty Python's Flying Circus.",
    full: "Monty Python were a British comedy troupe formed in 1969, best known for the " +
      "television series Monty Python's Flying Circus and for several feature films."
  },
  {
    id: 1090, title: "Mercury (planet)", redirects: [],
    extract: "Mercury is the smallest planet in the Solar System and the closest to the Sun.",
    full: "Mercury is the smallest planet in the Solar System and the closest to the Sun. It " +
      "completes an orbit every eighty-eight days and has almost no atmosphere, so its " +
      "surface temperature swings by hundreds of degrees between day and night."
  },
  {
    id: 1091, title: "Mercury (element)", redirects: [],
    extract: "Mercury is a chemical element with symbol Hg and atomic number 80, and the only " +
      "metal that is liquid at room temperature.",
    full: "Mercury is a chemical element with symbol Hg and atomic number 80, and the only " +
      "metal that is liquid at room temperature. It was used for centuries in thermometers " +
      "and barometers, and is now tightly controlled because its compounds are toxic."
  },
  {
    id: 1092, title: "Mercury", redirects: ["mercury"], disambiguation: true,
    extract: "Mercury may refer to the planet, the chemical element, or the Roman god of " +
      "commerce and messages.",
    full: "Mercury may refer to the planet, the chemical element, or the Roman god of commerce " +
      "and messages. See also: Mercury (mythology), Mercury Records."
  },
  {
    id: 1100, title: "Jaguar", redirects: ["jaguar"],
    extract: "The jaguar is a large cat native to the Americas and the third largest cat in " +
      "the world.",
    full: "The jaguar is a large cat native to the Americas and the third largest cat in the " +
      "world after the tiger and the lion. It has a powerful bite that lets it pierce the " +
      "shells of turtles, and its range once ran from the south-western United States to " +
      "Argentina."
  },
  {
    id: 1101, title: "Jaguar Cars", redirects: [],
    extract: "Jaguar Cars is a British luxury vehicle marque founded in 1935 and now part of " +
      "Jaguar Land Rover.",
    full: "Jaguar Cars is a British luxury vehicle marque founded in 1935 and now part of " +
      "Jaguar Land Rover. It is known for sports saloons and for a long history of endurance " +
      "racing."
  },
  {
    id: 1110, title: "Apple", redirects: ["apple"],
    extract: "An apple is the edible fruit of the apple tree, one of the most widely " +
      "cultivated tree fruits.",
    full: "An apple is the edible fruit of the apple tree, one of the most widely cultivated " +
      "tree fruits. Thousands of cultivars exist, selected for eating, cooking or cider. The " +
      "tree originated in Central Asia and has been grown in Europe and Asia for thousands of " +
      "years."
  },
  {
    id: 1111, title: "Apple Inc.", redirects: ["apple inc.", "apple inc"],
    extract: "Apple Inc. is an American technology company known for the Mac, the iPhone and " +
      "the iPad.",
    full: "Apple Inc. is an American technology company known for the Mac, the iPhone and the " +
      "iPad. It was founded in 1976 and designs both the hardware and the operating systems " +
      "its devices run."
  },
  {
    id: 1120, title: "Amazon rainforest", redirects: [],
    extract: "The Amazon rainforest is a moist broadleaf forest covering most of the Amazon " +
      "basin in South America.",
    full: "The Amazon rainforest is a moist broadleaf forest covering most of the Amazon basin " +
      "in South America. It holds the largest remaining tract of tropical rainforest on Earth " +
      "and an extraordinary concentration of species."
  },
  {
    id: 1121, title: "Amazon (company)", redirects: [],
    extract: "Amazon is an American multinational technology and retail company founded in " +
      "1994.",
    full: "Amazon is an American multinational technology and retail company founded in 1994. " +
      "It began as an online bookseller and now operates one of the largest cloud computing " +
      "businesses in the world."
  },
  {
    id: 1122, title: "Amazon River", redirects: [],
    extract: "The Amazon River is a river in South America and the largest river in the world " +
      "by discharge volume.",
    full: "The Amazon River is a river in South America and the largest river in the world by " +
      "discharge volume. It rises in the Andes and crosses the continent to the Atlantic."
  },
  {
    id: 1123, title: "Amazon", redirects: ["amazon"], disambiguation: true,
    extract: "Amazon may refer to the river, the rainforest, the company, or the warrior women " +
      "of Greek mythology.",
    full: "Amazon may refer to the river, the rainforest, the company, or the warrior women of " +
      "Greek mythology."
  },
  {
    id: 1130, title: "Saturn", redirects: ["saturn"],
    extract: "Saturn is the sixth planet from the Sun and the second largest in the Solar " +
      "System, known for its ring system.",
    full: "Saturn is the sixth planet from the Sun and the second largest in the Solar System, " +
      "known for its ring system. It is a gas giant with roughly ninety-five times the mass of " +
      "Earth and a mean density lower than that of water."
  },
  {
    id: 1131, title: "Saturn (rocket family)", redirects: [],
    extract: "Saturn was a family of American rockets developed for crewed spaceflight, " +
      "including the Saturn V used for the Apollo missions.",
    full: "Saturn was a family of American rockets developed for crewed spaceflight, including " +
      "the Saturn V used for the Apollo missions to the Moon."
  },
  {
    id: 1140, title: "Window", redirects: ["window"],
    extract: "A window is an opening in a wall or roof, usually fitted with glass, that admits " +
      "light and air.",
    full: "A window is an opening in a wall or roof, usually fitted with glass, that admits " +
      "light and air. Early windows were unglazed openings closed with shutters; glazing " +
      "became common in Europe from the seventeenth century."
  },
  {
    id: 1141, title: "Microsoft Windows", redirects: ["microsoft windows"],
    extract: "Microsoft Windows is a family of proprietary graphical operating systems " +
      "developed by Microsoft.",
    full: "Microsoft Windows is a family of proprietary graphical operating systems developed " +
      "by Microsoft. The first version shipped in 1985 as a graphical shell over MS-DOS."
  },
  {
    id: 1142, title: "Windows", redirects: ["windows"], disambiguation: true,
    extract: "Windows may refer to the Microsoft operating system family, or to the plural of " +
      "window.",
    full: "Windows may refer to the Microsoft operating system family, or to the plural of " +
      "window."
  },
  {
    id: 1150, title: "School", redirects: ["school"],
    extract: "A school is an institution designed to provide learning spaces and teaching " +
      "under the direction of teachers.",
    full: "A school is an institution designed to provide learning spaces and teaching under " +
      "the direction of teachers. Most countries have systems of formal education, which is " +
      "sometimes compulsory, and schooling is normally divided into stages by age."
  },
  {
    id: 1151, title: "School of Rock", redirects: [],
    extract: "School of Rock is a 2003 American comedy film starring Jack Black as a musician " +
      "who poses as a substitute teacher.",
    full: "School of Rock is a 2003 American comedy film starring Jack Black as a musician who " +
      "poses as a substitute teacher and forms a band from his class."
  },
  {
    id: 1160, title: "Matrix (mathematics)", redirects: [],
    extract: "In mathematics, a matrix is a rectangular array of numbers arranged in rows and " +
      "columns.",
    full: "In mathematics, a matrix is a rectangular array of numbers arranged in rows and " +
      "columns. Matrices represent linear maps between vector spaces once bases are chosen, " +
      "and matrix multiplication corresponds to composing those maps. They are central to " +
      "linear algebra, to computer graphics and to the numerical solution of systems of " +
      "equations."
  },
  {
    id: 1161, title: "The Matrix", redirects: ["the matrix"],
    extract: "The Matrix is a 1999 science fiction film written and directed by the " +
      "Wachowskis.",
    full: "The Matrix is a 1999 science fiction film written and directed by the Wachowskis, " +
      "in which a programmer discovers that his world is a simulation."
  },
  {
    id: 1162, title: "Matrix", redirects: ["matrix"], disambiguation: true,
    extract: "Matrix may refer to the mathematical object, the 1999 film, or a surrounding " +
      "medium in biology and materials science.",
    full: "Matrix may refer to the mathematical object, the 1999 film, or a surrounding medium " +
      "in biology and materials science."
  },
  {
    id: 1170, title: "Transformer", redirects: ["transformer"],
    extract: "A transformer is a passive electrical component that transfers energy between " +
      "circuits through electromagnetic induction.",
    full: "A transformer is a passive electrical component that transfers energy between " +
      "circuits through electromagnetic induction. Two or more coils share a magnetic core, " +
      "and the ratio of their turns sets the ratio of the voltages. Transformers make it " +
      "practical to move electrical power over long distances at high voltage and low current."
  },
  {
    id: 1171, title: "Transformer (deep learning architecture)", redirects: [],
    extract: "A transformer is a neural network architecture built around attention rather " +
      "than recurrence.",
    full: "A transformer is a neural network architecture built around attention rather than " +
      "recurrence. Each layer lets every position in a sequence attend to every other, which " +
      "removes the sequential bottleneck of earlier recurrent models."
  },
  {
    id: 1172, title: "Transformers (franchise)", redirects: [],
    extract: "Transformers is a media franchise built around transforming robot toys " +
      "introduced in 1984.",
    full: "Transformers is a media franchise built around transforming robot toys introduced " +
      "in 1984, and has since covered animation, comics and live-action films."
  },
  {
    id: 1180, title: "Rust", redirects: ["rust"],
    extract: "Rust is an iron oxide formed by the reaction of iron with oxygen in the presence " +
      "of water or moist air.",
    full: "Rust is an iron oxide formed by the reaction of iron with oxygen in the presence of " +
      "water or moist air. It is porous rather than protective, so corrosion continues " +
      "beneath the surface once it has started."
  },
  {
    id: 1181, title: "Rust (programming language)", redirects: [],
    extract: "Rust is a systems programming language that enforces memory safety without a " +
      "garbage collector.",
    full: "Rust is a systems programming language that enforces memory safety without a " +
      "garbage collector. Its ownership and borrowing rules are checked at compile time, so " +
      "whole classes of memory errors are rejected before a program runs."
  },
  {
    id: 1190, title: "Cell (biology)", redirects: [],
    extract: "A cell is the basic structural and functional unit of all known living " +
      "organisms.",
    full: "A cell is the basic structural and functional unit of all known living organisms. " +
      "It is bounded by a membrane and contains the machinery for metabolism and, in most " +
      "cases, a copy of the organism's genetic material."
  },
  {
    id: 1191, title: "Prison cell", redirects: [],
    extract: "A prison cell is a small room in a prison in which a prisoner is confined.",
    full: "A prison cell is a small room in a prison in which a prisoner is confined, usually " +
      "furnished with a bed and secured by a locked door."
  },
  {
    id: 1192, title: "Cell", redirects: ["cell"], disambiguation: true,
    extract: "Cell may refer to the biological unit, a prison room, an electrochemical cell, " +
      "or a spreadsheet entry.",
    full: "Cell may refer to the biological unit, a prison room, an electrochemical cell, or a " +
      "spreadsheet entry."
  },
  {
    id: 1200, title: "Quantum mechanics", redirects: ["quantum mechanics"],
    extract: "Quantum mechanics is the branch of physics that describes matter and energy at " +
      "the scale of atoms and subatomic particles.",
    full: "Quantum mechanics is the branch of physics that describes matter and energy at the " +
      "scale of atoms and subatomic particles. Quantities such as energy and angular momentum " +
      "take discrete values, a system is described by a wavefunction whose square gives a " +
      "probability, and certain pairs of properties cannot both be sharp at once."
  },
  {
    id: 1210, title: "Java coffee", redirects: [],
    extract: "Java coffee is coffee produced on the Indonesian island of Java.",
    full: "Java coffee is coffee produced on the Indonesian island of Java, historically " +
      "important enough that the island's name became a general slang term for coffee."
  }
];

const BY_ID = new Map(PAGES.map((p) => [p.id, p]));
const BY_TITLE = new Map(PAGES.map((p) => [p.title.toLowerCase(), p]));
const BY_REDIRECT = new Map();
PAGES.forEach((p) => (p.redirects || []).forEach((r) => BY_REDIRECT.set(r.toLowerCase(), p)));

function words(s) {
  return String(s || "").toLowerCase().match(/[a-z0-9]+/g) || [];
}

/* Deliberately naive lexical relevance -- title tokens weigh more than body
   tokens, shorter titles win ties. This is what makes "High School High" a
   plausible top hit for "high school", which is the failure the ladder exists
   to stop; a smarter fixture search would hide it. */
function search(query, limit) {
  const q = words(query);
  if (!q.length) return [];
  const scored = [];
  for (const p of PAGES) {
    const t = words(p.title), b = words(p.extract + " " + p.full);
    let title = 0, body = 0;
    for (const w of q) {
      if (t.includes(w)) title++;
      if (b.includes(w)) body++;
    }
    if (!title && !body) continue;
    const score = 3.2 * (title / q.length) + 0.9 * (body / q.length) -
      0.06 * Math.max(0, t.length - q.length);
    scored.push({ p, score });
  }
  scored.sort((a, b) => b.score - a.score || a.p.title.length - b.p.title.length);
  return scored.slice(0, limit || 3).map((s, i) => ({ page: s.p, index: i + 1 }));
}

function resolveTitle(title) {
  const key = String(title || "").trim().toLowerCase();
  if (!key) return null;
  if (BY_REDIRECT.has(key)) return { page: BY_REDIRECT.get(key), redirected: true };
  if (BY_TITLE.has(key)) return { page: BY_TITLE.get(key), redirected: false };
  return null;
}

module.exports = { PAGES, BY_ID, BY_TITLE, BY_REDIRECT, search, resolveTitle, words };
