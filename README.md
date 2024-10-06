
# Powehi - A Gamified Journey Through Space

Welcome to **Powehi**, an immersive web application where you explore the universe through stunning images captured by the **James Webb Space Telescope**. In this interactive, educational experience, you’ll navigate through space, unlock cosmic knowledge, and engage in quizzes to progress through the game. Designed for users of all backgrounds, Powehi makes learning about the universe both fun and accessible.

---

## Table of Contents

[Overview](#overview) [Game Flow](#game-flow) [Key Features](#key-features) [How to Play](#how-to-play) [Technical Details](#technical-details) [Credits](#credits) [Contact](#contact)

---

## Overview

**Powehi** combines space exploration with gamification to create an engaging experience. Using a simple scrolling mechanism, users will journey from Earth to the far reaches of space, accompanied by stunning visuals and educational facts about each cosmic object. At the end of each level, players must complete a quiz to unlock the next destination. Along the way, users are assisted by a character named *Sun*, who provides helpful explanations for complex concepts.



---

## Game Flow

1. **Introduction Page**  
   Start your journey with a brief introduction to the **James Webb Space Telescope** (JWST), its purpose, and how the game works.
   
2. **Scrolling Through Space**  
   The core gameplay involves scrolling through space as you move between 12 breathtaking images captured by JWST. With each scroll, you'll encounter fascinating information about the objects you see.

3. **Guided Learning with Sun**  
   A character named *Sun* is your guide, offering clarifications on scientific terms and enhancing your understanding of the cosmos.

4. **Atmospheric Sound Design**  
   The ambient space sounds immerse you in the game. As you approach each image, the background music changes, created through unique audio generated from the image data itself.

5. **Quizzes**  
   After reaching each image, a quiz question awaits. Answer correctly to move to the next level. Unlimited attempts encourage learning and exploration.

6. **Conclusion**  
   The game concludes with a deep field image from JWST, marking the end of your cosmic journey.

7. **Leaderboard**  
   Players are ranked based on how quickly they complete the game. Compete against friends or challenge yourself to improve your time.

8. **Archive Access**  
   After completing the game, unlock an archive of over 500 images from JWST, each with its own unique music and descriptive content.

---

## Key Features

- **Educational Support**: Learn about space through detailed facts about each image and additional help from *Sun* for complex terminology.
- **Engaging Quizzes**: Test your knowledge with quizzes at the end of each level. Unlimited attempts allow you to learn at your own pace.
- **Immersive Soundscapes**: Experience space in a new way with music generated from each image, adding an auditory dimension to the visuals.
- **Leaderboard**: Track your progress and compete with others based on how fast you complete the game.
- **Archive of Images**: After finishing the game, explore a collection of 500+ JWST images with custom music and detailed information on how each piece was created.

---

## How to Play

1. **Start**: Begin on the introduction page, where you’ll learn the basics of the **James Webb Space Telescope** and the game’s instructions.
2. **Scroll**: Scroll through space from one image to the next. As you move, read facts about each celestial object.
3. **Learn with Sun**: If a term is unfamiliar, *Sun* will help you understand it with clear explanations.
4. **Answer the Quiz**: At the end of each level, answer a quiz question to proceed. You can try as many times as needed to get it right.
5. **Explore the Archive**: After completing the game, explore a vast archive of JWST images, each paired with music and detailed descriptions.

---

## Technical Details

- **Language**: The game is built using modern web technologies such as **HTML**, **CSS**, and **JavaScript** for an interactive experience.
- **Music Generation**: The music in the game is created by converting visual data from JWST images into audio components.  
   - **Image Processing**: Utilizes **OpenCV** and **PIL (Python Imaging Library)** for image manipulation.
   - **Music Mapping**: Converts color, brightness, and other visual features into musical elements like tempo, key, and rhythm.
   - **MIDI Creation**: The **midiutil** library generates MIDI files based on the processed image data, resulting in a unique soundtrack for each image.
- **Responsive Design**: Powehi is optimized for desktop and mobile devices, ensuring an immersive experience across all platforms.

---

## Credits

- **Development Team**: [Your Name], [Team Members]
- **Music Generation**: [Contributor Names]
- **Design and UI**: [Designer Names]
- **Special Thanks**: [Contributors, Advisors]

---

## Contact

If you have any questions, feedback, or issues, feel free to contact us at:

- **Email**: [your-email@example.com]
- **Website**: [your-website.com]  
