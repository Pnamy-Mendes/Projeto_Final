import React from 'react';

function Carousel({ history }) {
    return (
        <section>
            <div className="container">
                <div className="carousel">
                    {history.map((item, index) => (
                        <div key={index} className="carousel__slide">
                            <figure>
                                <div>
                                    <img src={item.image_path} alt={`Mood: ${item.mood} (${item.confidence}%)`} />
                                </div>
                                <figcaption>
                                    Mood: {item.mood} ({item.confidence}%)
                                </figcaption>
                            </figure>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}

export default Carousel;
